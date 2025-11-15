# main.py
"""
Main script to run the insulin transport simulation and generate figures.
Run with:
    python main.py
"""

import numpy as np

import parameters as prm
from simulation import run_simulation
from visualization import plot_snapshot, plot_centerline_profile


def main():
    snapshots, times = run_simulation()
    times_arr = np.array(times)

    # 1) initial snapshot (t ≈ 0)
    idx0 = 0

    # 2) when advective front reaches roughly x = L/2
    L_half = 0.5 * prm.L                 # m
    t_front_mid = L_half / prm.v_blood   # s  (≈ 0.125 s)
    idx_front_mid = int(np.argmin(np.abs(times_arr - t_front_mid)))

    # 3) middle of total simulated time
    idx_mid = len(times_arr) // 2

    # 4) final snapshot
    idx_final = len(times_arr) - 1

    indices = [idx0, idx_front_mid, idx_mid, idx_final]
    labels  = ["t0", "t_frontMid", "tmid", "tfinal"]

    for idx, label in zip(indices, labels):
        C = snapshots[idx]
        t = times[idx]

        # 2D heatmap
        plot_snapshot(
            C, prm.x, prm.y, t,
            filename=f"snapshot_{label}.png",
            show=False
        )

        # 1D profile along capillary centerline
        plot_centerline_profile(
            C, prm.x, prm.y, t,
            filename=f"profile_{label}.png",
            show=False
        )

    print("Simulation complete. Figures saved in 'figures/'.")


if __name__ == "__main__":
    main()