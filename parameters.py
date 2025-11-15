# parameters.py
import numpy as np

# ----- Geometry (in meters) -----
L = 200e-6   # 200 micrometers in x (length)
H = 100e-6   # 100 micrometers in y (height)
h_blood = 10e-6  # blood layer thickness (capillary lumen)

nx = 120
ny = 60

dx = L / (nx - 1)
dy = H / (ny - 1)

x = np.linspace(0.0, L, nx)
y = np.linspace(0.0, H, ny)

# index of vessel wall (blood–tissue interface)
j_wall = int(h_blood / dy)

# ----- Physical parameters (example values – to refine from literature) -----
D_blood   = 1.0e-10    # m^2/s, insulin diffusion in blood/plasma
D_tissue  = 5.0e-11    # m^2/s, insulin diffusion in muscle tissue (slower)
v_blood   = 800e-6     # m/s, axial blood velocity (800 µm/s)
k_uptake  = 0.1        # 1/s, first-order uptake rate in tissue (toy value)

C_in      = 1.0        # arbitrary insulin concentration at inlet (nondimensional or mol/m^3)

# ----- Time stepping parameters -----
# Diffusion stability estimate for 2D explicit FD
D_max = max(D_blood, D_tissue)
dt_diff = 0.5 / (2 * D_max * (1.0/dx**2 + 1.0/dy**2))

# Advection CFL condition: v * dt / dx <= 1
dt_adv = dx / v_blood

# Choose dt smaller than both
dt = 0.2 * min(dt_diff, dt_adv)   # safety factor 0.2

t_final = 20.0   # total simulated time [s] (adjust as you wish)
n_steps = int(t_final / dt)

# How often to save snapshots for plotting (every N time steps)
save_every = 10