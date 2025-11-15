# animate.py
"""
Generate an animation of the insulin transport simulation.

Run with:
    python animate.py
"""
# animate.py
import parameters as prm
from simulation import run_simulation
from visualization import animate_snapshots


def main():
    snapshots, times = run_simulation()
    print(f"Number of frames in animation: {len(snapshots)}")
    print(f"Time range: {times[0]:.3f} s → {times[-1]:.3f} s")

    animate_snapshots(
        snapshots,
        prm.x,
        prm.y,
        times,
        filename="insulin_capillary_animation.mp4",
        fps=200  # you can try 3, 5, 8 etc.
    )

    print("Done. Animation saved as 'insulin_capillary_animation.mp4'.")


if __name__ == "__main__":
    main()