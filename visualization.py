# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import parameters as prm
from matplotlib import animation


FIG_DIR = "figures"


def ensure_fig_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)


def plot_snapshot(C, x, y, t, filename=None, show=False):
    """
    Plot a 2D heatmap of concentration C(x,y) at time t.
    """
    ensure_fig_dir()

    X, Y = np.meshgrid(x*1e6, y*1e6)  # convert to micrometers for labels

    plt.figure(figsize=(6, 3))
    im = plt.pcolormesh(X, Y, C, shading='auto')
    plt.colorbar(im, label="Insulin concentration (arb. units)")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(f"Insulin concentration at t = {t:.2f} s")

    if filename is not None:
        path = os.path.join(FIG_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"Saved figure: {path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_centerline_profile(C, x, y, t, filename=None, show=False):
    """
    Plot concentration along the vessel axis at the mid-blood height.
    """
    ensure_fig_dir()

    # choose a representative row inside the blood region
    j_center_blood = max(0, min(prm.j_wall, prm.j_wall // 2))
    C_line = C[j_center_blood, :]

    plt.figure(figsize=(5, 3))
    plt.plot(x*1e6, C_line, '-o', markersize=2)
    plt.xlabel("x (µm)")
    plt.ylabel("Insulin concentration (arb. units)")
    plt.title(f"Centerline concentration at t = {t:.2f} s")
    
    if filename is not None:
        path = os.path.join(FIG_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"Saved figure: {path}")

    if show:
        plt.show()
    else:
        plt.close()



def animate_snapshots(
    snapshots, x, y, times,
    filename="insulin_capillary_animation.mp4",
    fps=5
):
    """
    Create an animation of the 2D concentration field over time.

    Parameters
    ----------
    snapshots : list of 2D arrays
        C(x,y) fields saved at different times.
    x, y : 1D arrays
        Spatial coordinates (in meters).
    times : list of float
        Times corresponding to each snapshot (in seconds).
    filename : str
        Output file name (.mp4 or .gif).
    fps : int
        Frames per second for the saved video.
    """
    snapshots_arr = np.array(snapshots)   # (nt, ny, nx)
    nt, ny, nx = snapshots_arr.shape

    # Fix color scale across frames
    vmin = snapshots_arr.min()
    vmax = snapshots_arr.max()

    # Axes in micrometers
    extent = [x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6]

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        snapshots_arr[0],
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Insulin concentration (a.u.)")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    title = ax.set_title(f"t = {times[0]:.1f} s")

    def update(frame_idx):
        im.set_data(snapshots_arr[frame_idx])
        title.set_text(f"t = {times[frame_idx]:.1f} s")
        return im, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=1000 / fps,  # ms between frames
        blit=False
    )

    print(f"Saving animation to {filename} ...")

    if filename.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=200)
    else:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)

    plt.close(fig)
    print("Animation saved.")