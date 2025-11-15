# tests.py
"""
Sanity / verification tests for the insulin capillary project.

Each test isolates one physical process:
  1) Diffusion only          (2D)
  2) Reaction (uptake) only  (0D, spatially uniform)
  3) Advection only          (1D along x)
  4) Advection only          (2D Gaussian blob advected along x)

Run with:
    python tests.py
"""

import numpy as np
import parameters as prm
from visualization import plot_snapshot


# ---------- Helper: Neumann BC (zero-flux on all sides) ----------

def apply_neumann_all(C):
    """
    Apply zero-normal-flux (Neumann) boundary conditions on all sides:
      dC/dn = 0  => copy interior value to boundary.
    Works in-place and also returns C for convenience.
    """
    # bottom and top
    C[0, :]  = C[1, :]
    C[-1, :] = C[-2, :]
    # left and right
    C[:, 0]  = C[:, 1]
    C[:, -1] = C[:, -2]
    return C


# ================================================================
# TEST 1: PURE DIFFUSION (2D)
# ================================================================

def diffusion_step(C, D, dx, dy, dt):
    """
    One explicit time step for pure diffusion:
        dC/dt = D * (d2C/dx2 + d2C/dy2)
    using central differences for second derivatives.
    Neumann BCs are applied outside this function.
    """
    Cn = C.copy()
    ny, nx = Cn.shape

    d2Cdx2 = np.zeros_like(Cn)
    d2Cdy2 = np.zeros_like(Cn)

    # second derivative in x
    d2Cdx2[:, 1:-1] = (Cn[:, 2:] - 2.0*Cn[:, 1:-1] + Cn[:, 0:-2]) / dx**2
    d2Cdx2[:, 0]    = d2Cdx2[:, 1]
    d2Cdx2[:, -1]   = d2Cdx2[:, -2]

    # second derivative in y
    d2Cdy2[1:-1, :] = (Cn[2:, :] - 2.0*Cn[1:-1, :] + Cn[0:-2, :]) / dy**2
    d2Cdy2[0, :]    = d2Cdy2[1, :]
    d2Cdy2[-1, :]   = d2Cdy2[-2, :]

    C_new = Cn + dt * D * (d2Cdx2 + d2Cdy2)
    return C_new


def run_test_diffusion():
    """
    Start with a small 'bump' in the middle of the domain and let it diffuse.
    With only diffusion and Neumann BCs, total mass should be ~constant.
    """
    ny, nx = prm.ny, prm.nx

    # initial concentration: small square bump in the center
    C = np.zeros((ny, nx))
    j_mid = ny // 2
    i_mid = nx // 2
    C[j_mid-1:j_mid+2, i_mid-1:i_mid+2] = 1.0

    # uniform diffusion coefficient
    D = prm.D_tissue  # arbitrary choice

    # choose a shorter test time
    t_final = 1.0  # seconds
    n_steps = int(t_final / prm.dt)

    print("=== TEST 1: PURE DIFFUSION (2D) ===")
    print(f"Using D = {D:.2e}, dt = {prm.dt:.2e}, steps = {n_steps}")

    # initial mass
    apply_neumann_all(C)
    initial_mass = C.sum() * prm.dx * prm.dy

    for n in range(n_steps):
        C = diffusion_step(C, D, prm.dx, prm.dy, prm.dt)
        C = apply_neumann_all(C)

        if n % max(1, n_steps // 10) == 0 or n == n_steps - 1:
            t = (n+1) * prm.dt
            total_mass = C.sum() * prm.dx * prm.dy
            print(f"t = {t:.3f} s, total mass = {total_mass:.6e}")

    print(f"Initial mass: {initial_mass:.6e}")
    print("If diffusion is implemented correctly with Neumann BCs,")
    print("the total mass should stay almost constant (small numerical drift is OK).")

    # save final snapshot
    plot_snapshot(C, prm.x, prm.y, t_final, filename="test_diffusion_final.png")


# ================================================================
# TEST 2: PURE REACTION (UPTAKE)
# ================================================================

def run_test_reaction():
    """
    Test the uptake term: dC/dt = -k C, no spatial dependence.
    Analytic solution for uniform initial condition C0 is: C(t) = C0 * exp(-k t).
    We compare the mean concentration to this analytic solution.
    """
    ny, nx = prm.ny, prm.nx

    C0 = 1.0
    C = np.ones((ny, nx)) * C0

    k = prm.k_uptake  # uniform uptake rate

    t_final = 5.0  # seconds
    dt = prm.dt
    n_steps = int(t_final / dt)

    print("\n=== TEST 2: PURE REACTION (UPTAKE) ===")
    print(f"Using k = {k:.2e}, dt = {dt:.2e}, steps = {n_steps}")

    for n in range(n_steps):
        # explicit Euler for ODE: dC/dt = -k C
        C = C + dt * (-k * C)
        t = (n+1) * dt

        if n % max(1, n_steps // 10) == 0 or n == n_steps - 1:
            mean_C = C.mean()
            exact = C0 * np.exp(-k * t)
            print(f"t = {t:.2f} s, mean C = {mean_C:.4f}, exact = {exact:.4f}")

    print("If the uptake term and time stepping are correct,")
    print("the mean concentration should follow exp(-k t) fairly closely.")


# ================================================================
# TEST 3: PURE ADVECTION (1D)
# ================================================================

def advection_step_1d(C, v, dx, dt):
    """
    1D upwind scheme for dC/dt + v dC/dx = 0 with v >= 0.
    Neumann BC at the right boundary (copy from the last interior point).
    """
    Cn = C.copy()
    nx = len(Cn)

    dCdx = np.zeros_like(Cn)
    # upwind: dC/dx ≈ (C_i - C_{i-1}) / dx
    dCdx[1:] = (Cn[1:] - Cn[:-1]) / dx
    dCdx[0]  = dCdx[1]

    C_new = Cn - dt * v * dCdx

    # Neumann at right boundary
    C_new[-1] = C_new[-2]

    return C_new


def run_test_advection_1d():
    """
    1D advection test along the x direction.
    Start with a bump near the inlet and see if it moves right
    at approximately speed v without exploding.
    """
    nx = prm.nx
    dx = prm.dx
    v  = prm.v_blood  # positive velocity

    # 1D concentration along x
    C = np.zeros(nx)
    C[1:4] = 1.0  # initial bump near inlet

    t_final = 0.2  # seconds
    dt = min(prm.dt, 0.5 * dx / v)  # satisfy CFL
    n_steps = int(t_final / dt)

    print("\n=== TEST 3: PURE ADVECTION (1D) ===")
    print(f"Using v = {v:.2e}, dx = {dx:.2e}, dt = {dt:.2e}, steps = {n_steps}")
    print("The bump should move to the right over time without blowing up.")

    for n in range(n_steps):
        C = advection_step_1d(C, v, dx, dt)
        t = (n+1) * dt

        if n % max(1, n_steps // 5) == 0 or n == n_steps - 1:
            x_pos = np.sum(C * np.arange(nx)) / (np.sum(C) + 1e-16)
            phys_x = x_pos * dx * 1e6  # micrometers
            print(f"t = {t:.4f} s, approximate bump center at x ≈ {phys_x:.2f} µm")

    # For a quick visual, embed C into a 2D field and save a snapshot
    C2D = np.tile(C, (prm.ny, 1))
    plot_snapshot(C2D, prm.x, prm.y, t_final, filename="test_advection1d_final.png")


# ================================================================
# TEST 4: PURE ADVECTION (2D GAUSSIAN BLOB)
# ================================================================

def advection_step_2d_x(C, u, dx, dt):
    """
    2D advection in x-direction only:
        dC/dt + u dC/dx = 0
    using an upwind scheme in x and Neumann BCs applied outside.
    """
    Cn = C.copy()
    ny, nx = Cn.shape

    dCdx = np.zeros_like(Cn)
    # upwind in x for all rows
    dCdx[:, 1:] = (Cn[:, 1:] - Cn[:, 0:-1]) / dx
    dCdx[:, 0]  = dCdx[:, 1]

    C_new = Cn - dt * u * dCdx
    return C_new


def run_test_advection_2d():
    """
    2D advection test:
    A Gaussian blob is initialized in the interior of the domain and advected
    to the right with constant velocity u. The blob should translate without
    blowing up, and the center of mass should move approximately as x(t) = x0 + u t.
    """
    ny, nx = prm.ny, prm.nx
    dx, dy = prm.dx, prm.dy

    # use the same x,y grids as the main simulation
    X, Y = np.meshgrid(prm.x, prm.y)

    # initial blob center and width
    x0 = 0.25 * prm.L       # 1/4 of vessel length
    y0 = 0.5 * prm.H        # mid-height
    sigma = 20e-6           # 20 µm width

    # Gaussian blob
    C = np.exp(-(((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2)))

    # advection speed (x-direction)
    u = prm.v_blood  # reuse blood velocity

    # choose final time so blob stays inside domain:
    # move it by about L/4
    t_final = 0.25 * prm.L / u
    dt = min(prm.dt, 0.5 * dx / u)  # CFL
    n_steps = int(t_final / dt)

    print("\n=== TEST 4: PURE ADVECTION (2D) ===")
    print(f"Using u = {u:.2e} m/s, t_final = {t_final:.3f} s, dt = {dt:.2e}, steps = {n_steps}")

    for n in range(n_steps):
        C = advection_step_2d_x(C, u, dx, dt)
        C = apply_neumann_all(C)
        t = (n+1) * dt

        if n % max(1, n_steps // 5) == 0 or n == n_steps - 1:
            # center of mass in x
            x_cm = (C * X).sum() / (C.sum() + 1e-16)
            x_exact = x0 + u * t
            print(f"t = {t:.4f} s, x_cm = {x_cm*1e6:6.2f} µm, exact ≈ {x_exact*1e6:6.2f} µm")

    # save final 2D snapshot of the advected blob
    plot_snapshot(C, prm.x, prm.y, t_final, filename="test_advection2d_final.png")


# ================================================================
# MAIN
# ================================================================

def main():
    run_test_diffusion()
    run_test_reaction()
    run_test_advection_1d()
    run_test_advection_2d()
    print("\nAll tests finished. See printed numbers and figures in 'figures/'.")


if __name__ == "__main__":
    main()