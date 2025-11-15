# **Insulin Transport in a Capillary–Tissue System**  
*A 2D finite-difference simulation of advection, diffusion, and cellular uptake*

This project implements a numerical simulation of insulin transport from a capillary into surrounding tissue. It models **advection** (blood flow inside the vessel), **diffusion** (spreading into tissue), and **reaction/uptake** (consumption by tissue), using a fully explicit finite-difference scheme in Python.

The goal is to visualize how insulin moves through a microvessel, spreads into tissue, and decays over time, while keeping the code modular and physically interpretable.

---

## **Features**
- 2D finite-difference solver for:
  - **Advection** inside a capillary  
  - **Diffusion** in both capillary and tissue  
  - **First-order uptake reaction** in tissue  
- Fully explicit update scheme  
- Neumann boundary conditions (zero-flux) on all sides  
- Visualization tools:
  - 2D heatmaps (snapshots)
  - 1D centerline concentration profiles
  - Full animation of insulin transport  
- Modular code: `parameters.py`, `simulation.py`, `visualization.py`, `animate.py`
- Optional physics verification tests (`tests.py`) for diffusion, reaction, and advection.

---

## **Project Structure**

```
Insulin-Capillary-Project/
│
├── main.py                 # Run the full simulation and save figures
├── animate.py              # Create an animation from saved snapshots
├── parameters.py           # Physical and numerical constants
├── simulation.py           # Core PDE solver
├── visualization.py        # Plotting routines
├── tests.py                # Physics sanity tests
│
└── figures/                # Saved snapshots, profiles, and animations
```

---

## **Physical Model**

We solve:

$begin:math:display$
\\frac\{\\partial C\}\{\\partial t\}
\= \- \\mathbf\{v\} \\cdot \\nabla C 
\+ D \\nabla\^2 C
\- kC
$end:math:display$

Where:

- $begin:math:text$ \\mathbf\{v\} \= \(v\_x\, 0\) $end:math:text$ is blood flow velocity (nonzero inside capillary, zero in tissue)  
- $begin:math:text$ D $end:math:text$ is diffusion coefficient  
- $begin:math:text$ k $end:math:text$ is first-order uptake rate in tissue  
- Boundary conditions:  
  $begin:math:display$
  \\frac\{\\partial C\}\{\\partial n\} \= 0 \\quad \\text\{on all sides\}
  $end:math:display$

Initial condition:

- Insulin bolus placed at inlet region of capillary.

The tissue is represented as a 2D rectangle; the capillary is a thin horizontal band.

---

## **Generated Figures**

### **Snapshots (2D Heatmaps)**
- insulin distribution at:
  - $begin:math:text$ t \= 0 $end:math:text$
  - front-mid (when insulin reaches mid-capillary)
  - mid-simulation
  - final time  

### **Profiles (1D centerline cuts)**
- concentration along the central horizontal line of the capillary  
- useful to observe:
  - advection-dominated front shape  
  - smoothing by diffusion  
  - exponential decay from uptake

---

## **Animation**

To generate a full MP4 animation:

```bash
python animate.py
```

Requirements:
- **ffmpeg** installed  
- adjustable parameters:
  - `save_every` in `parameters.py` (controls number of saved frames)
  - `fps` in `animate.py` (controls animation duration)

---

## **Physics Tests**

You can verify the correctness of individual processes:

```bash
python tests.py
```

Included tests:
1. **Pure diffusion**  
   - mass should remain constant
2. **Pure reaction**  
   - decay matches $begin:math:text$ C\(t\) \= C\_0 e\^\{\-kt\} $end:math:text$
3. **Pure 1D advection**  
   - a bump moves with speed $begin:math:text$ v $end:math:text$ without numerical blowup

These tests use stripped-down grids and simplified physics for clarity.

---

## **Run the Simulation**

```bash
python main.py
```

This will create:

- 4 heatmap snapshots  
- 4 centerline profiles  
- saved to `figures/`

All parameters (geometry, diffusion, uptake rate, velocity, numerical grid) are configured inside **`parameters.py`**.

---

## **Dependencies**
- Python 3.9+
- NumPy
- Matplotlib
- Pillow (for fallback animation writer)
- FFmpeg (optional, for MP4 output)

Install missing packages:

```bash
pip install numpy matplotlib pillow
```

If ffmpeg is missing:

```bash
brew install ffmpeg     # macOS
sudo apt-get install ffmpeg   # Ubuntu / Linux
```

---

## **License**

MIT License — free, open, shareable.

---

## **Notes**

This repository contains the simulation code developed for the **GBE361 – Biotransport Phenomena** course at **Yeditepe University, Istanbul, Turkey**.

The project models **insulin transport from a capillary into surrounding tissue**, using a 2D advection–diffusion–reaction equation solved with finite differences. It is designed as an **educational, visualizable model** of biochemical transport. It is not intended to reproduce exact physiological values, but to explore the interplay between advection, diffusion, and uptake in a controlled environment.

---


### Use of AI Assistance
During the development of this project, I used **AI tools (ChatGPT)** to:
- troubleshoot numerical stability issues,  
- refine finite-difference schemes,  
- generate test scripts for verification,  
- improve code organization and documentation.

All scientific decisions, modeling choices, parameter selection, and final code implementation were **reviewed, verified, and approved by me**.

The project reflects **my own work**, with AI used as a programming and conceptual assistant — similar to using StackOverflow or scientific discussion tools.