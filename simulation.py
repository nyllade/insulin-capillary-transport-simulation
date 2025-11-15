# simulation.py
"""
Core simulation routines for the insulin transport model.
Includes initialization, time-stepping, and the main simulation loop.
"""

import numpy as np
import parameters as prm


def initialize_fields():
    """
    Create initial concentration field C and parameter fields D, k, vx.

    Returns
    -------
    C : (ny, nx) array
        Initial insulin concentration (all zeros).
    D : (ny, nx) array
        Diffusion coefficient field.
    k : (ny, nx) array
        Uptake rate field.
    vx : (ny, nx) array
        Axial velocity field.
    """
    ny, nx = prm.ny, prm.nx

    # concentration
    C = np.zeros((ny, nx))

    # parameter fields
    D  = np.zeros((ny, nx))
    k  = np.zeros((ny, nx))
    vx = np.zeros((ny, nx))

    # blood region: rows 0..j_wall
    D[0:prm.j_wall+1, :]  = prm.D_blood
    k[0:prm.j_wall+1, :]  = 0.0
    vx[0:prm.j_wall+1, :] = prm.v_blood

    # tissue region: rows j_wall+1..ny-1
    D[prm.j_wall+1:, :] = prm.D_tissue
    k[prm.j_wall+1:, :] = prm.k_uptake
    vx[prm.j_wall+1:, :] = 0.0

    return C, D, k, vx


def apply_boundary_conditions(C):
    """
    Apply inlet, outlet, and top/bottom boundary conditions to C in-place.

    BCs:
      - inlet (x=0, blood rows): Dirichlet C = C_in
      - inlet (x=0, tissue rows): zero-gradient in x
      - outlet (x=L): zero-gradient in x
      - bottom (y=0): zero-flux in y
      - top (y=H): zero-flux in y
    """
    ny, nx = C.shape

    # --- inlet x = 0 ---
    # blood part: rows 0..j_wall -> fixed C_in
    C[0:prm.j_wall+1, 0] = prm.C_in
    # tissue part at x=0: copy from neighbor (zero-gradient)
    C[prm.j_wall+1:, 0] = C[prm.j_wall+1:, 1]

    # --- outlet x = L (zero-gradient) ---
    C[:, -1] = C[:, -2]

    # --- bottom y=0 (zero-flux) ---
    C[0, :] = C[1, :]

    # --- top y=H (zero-flux) ---
    C[-1, :] = C[-2, :]

    return C


def step(C, D, k, vx, dx, dy, dt):
    """
    Perform one explicit time step update of the concentration field C.

    Uses:
      - upwind scheme for advection in x
      - central differences for diffusion in x and y
      - explicit first-order uptake term -k*C
    """
    Cn = C.copy()
    ny, nx = Cn.shape

    # --- Derivatives arrays ---
    dCdx    = np.zeros_like(Cn)
    d2Cdx2  = np.zeros_like(Cn)
    d2Cdy2  = np.zeros_like(Cn)

    # ---- x-derivative: upwind for positive vx ----
    # We assume vx >= 0 in blood region, 0 in tissue.
    # dCdx ≈ (C_i - C_{i-1}) / dx for i >= 1
    dCdx[:, 1:] = (Cn[:, 1:] - Cn[:, 0:-1]) / dx
    dCdx[:, 0]  = dCdx[:, 1]  # just copy (will be overridden by BCs)

    # ---- second derivative in x: central difference ----
    d2Cdx2[:, 1:-1] = (Cn[:, 2:] - 2.0*Cn[:, 1:-1] + Cn[:, 0:-2]) / dx**2
    d2Cdx2[:, 0]    = d2Cdx2[:, 1]
    d2Cdx2[:, -1]   = d2Cdx2[:, -2]

    # ---- second derivative in y: central difference ----
    d2Cdy2[1:-1, :] = (Cn[2:, :] - 2.0*Cn[1:-1, :] + Cn[0:-2, :]) / dy**2
    d2Cdy2[0, :]    = d2Cdy2[1, :]
    d2Cdy2[-1, :]   = d2Cdy2[-2, :]

    # --- PDE terms ---
    adv_term   = -vx * dCdx                    # advection
    diff_term  = D * (d2Cdx2 + d2Cdy2)         # diffusion
    react_term = -k * Cn                       # uptake (reaction)

    # explicit Euler update
    C_new = Cn + dt * (adv_term + diff_term + react_term)

    # re-apply boundary conditions
    C_new = apply_boundary_conditions(C_new)

    return C_new


def run_simulation():
    """
    Run the full time integration and return snapshots for visualization.

    Returns
    -------
    snapshots : list of 2D arrays
        Saved concentration fields at selected times.
    times : list of float
        Times corresponding to each snapshot.
    """
    C, D, k, vx = initialize_fields()
    C = apply_boundary_conditions(C)  # enforce BCs at t=0

    snapshots = []
    times = []

    # --- compute the time index when the front should be around x = L/2 ---
    #    t_frontMid ≈ (L/2) / v_blood
    t_front_mid = 0.5 * prm.L / prm.v_blood          # seconds
    n_front_mid = int(round(t_front_mid / prm.dt))   # time-step index

    for n in range(prm.n_steps):
        t = n * prm.dt

        # take one step
        C = step(C, D, k, vx, prm.dx, prm.dy, prm.dt)

        # save snapshots:
        # - regularly every save_every steps
        # - always at the "front-mid" step n_front_mid
        # - and at the final step
        if (n % prm.save_every == 0) or (n == prm.n_steps - 1) or (n == n_front_mid):
            snapshots.append(C.copy())
            times.append(t)

    return snapshots, times