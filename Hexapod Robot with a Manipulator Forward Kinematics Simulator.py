import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 0.2, 0.2

# Manipulator lengths
Lm1, Lm2, Lm3 = 0.02, 0.2, 0.2

right_legs = ["FR", "MR", "RR"]
left_legs  = ["FL", "ML", "RL"]

limits = {
    "right": {
        "theta1": (-np.pi/6, np.pi/6),
        "theta2": (-np.pi/6,  np.pi/2),
        "theta3": (-np.pi,          0)
    },
    "left": {
        "theta1": (-np.pi/6,  np.pi/6),
        "theta2": (-np.pi/2,  np.pi/6),
        "theta3": (0,         np.pi)
    }
}

# Manipulator limits
manip_limits = [
    (-np.pi, np.pi),
    (0, np.pi),
    (-np.pi, np.pi)
]

# Six legs
legs = {
    "FR": [0.0, 0.5, -1.0],
    "FL": [0.0, -0.5, 1.0],
    "MR": [0.0, 0.5, -1.0],
    "ML": [0.0, -0.5, 1.0],
    "RR": [0.0, 0.5, -1.0],
    "RL": [0.0, -0.5, 1.0]
}

# -----------------------------
# Manipulator state
# -----------------------------
manip = [0.0, np.pi/3, -np.pi/3]
manip_default = manip[:]

# Store defaults
defaults = {k: v[:] for k, v in legs.items()}

# -----------------------------
# Hexagon body geometry
# -----------------------------
hex_radius = 0.35
angles = np.deg2rad([0, 60, 120, 180, 240, 300])
body = np.array([[hex_radius*np.cos(a), hex_radius*np.sin(a), 0] for a in angles] +
                [[hex_radius*np.cos(angles[0]), hex_radius*np.sin(angles[0]), 0]])

base_pos = {
    "FR": body[5],
    "FL": body[4],
    "ML": body[3],
    "MR": body[0],
    "RL": body[2],
    "RR": body[1]
}

# -----------------------------
# FK (legs)
# -----------------------------
def fk(t1, t2, t3):
    X = L1*np.sin(t2) + L2*np.sin(t2+t3)
    Y = (L1*np.cos(t2) + L2*np.cos(t2+t3)) * np.sin(t1)
    Z = -(L1*np.cos(t2) + L2*np.cos(t2+t3)) * np.cos(t1)
    return np.array([X, Y, Z])

# -----------------------------
# FK (Manipulator)
# -----------------------------
def fk_manip(t1, t2, t3):
    X = (Lm2*np.cos(t2) + Lm3*np.cos(t2+t3)) * np.sin(t1)
    Y = (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.cos(t1)
    Z = Lm2*np.sin(t2) + Lm3*np.sin(t2+t3) + Lm1
    return np.array([X, Y, Z])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

def draw():
    ax.clear()
    ax.plot(body[:,0], body[:,1], body[:,2], linewidth=3)

    # Draw legs
    for leg, (t1, t2, t3) in legs.items():
        base = base_pos[leg]

        knee = base + np.array([
            L1*np.sin(t2),
            L1*np.cos(t2)*np.sin(t1),
            -L1*np.cos(t2)*np.cos(t1)
        ])

        foot = base + fk(t1, t2, t3)

        ax.plot([base[0], knee[0]], [base[1], knee[1]], [base[2], knee[2]], marker='o')
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], [knee[2], foot[2]], marker='o')

    # -----------------------------
    # Draw Manipulator
    # -----------------------------
    base = np.array([0.0, 0.22, 0.0])

    t1, t2, t3 = manip

    j1 = base + np.array([0, 0, Lm1])
    j2 = j1 + np.array([
        Lm2 * np.cos(t2) * np.sin(t1),
        Lm2*np.cos(t2)*np.cos(t1),
        Lm2*np.sin(t2)
    ])
    end = base + fk_manip(t1, t2, t3)

    ax.plot([base[0], j1[0]], [base[1], j1[1]], [base[2], j1[2]], marker='o')
    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], marker='o')
    ax.plot([j2[0], end[0]], [j2[1], end[1]], [j2[2], end[2]], marker='o')

    ax.set_xlim([-0.7,0.7])
    ax.set_ylim([-0.7,0.7])
    ax.set_zlim([-0.7,0.7])

# -----------------------------
# Tkinter UI
# -----------------------------
root = tk.Tk()
root.title("Hexapod FK Simulator")
root.geometry("1200x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control = tk.Frame(root)
control.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# -----------------------------
# Update
# -----------------------------
def update(leg, i, val):
    legs[leg][i] = float(val)
    draw()
    canvas.draw_idle()

def update_manip(i, val):
    manip[i] = float(val)
    draw()
    canvas.draw_idle()

sliders = []
manip_sliders = []

# -----------------------------
# Leg sliders
# -----------------------------
for leg in legs:
    f = tk.LabelFrame(control, text=leg)
    f.pack(fill="x", pady=6)

    group = "right" if leg in right_legs else "left"

    for i, joint_name in enumerate(["theta1","theta2","theta3"]):
        min_lim, max_lim = limits[group][joint_name]

        s = tk.Scale(
            f,
            from_=min_lim,
            to=max_lim,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label=f"θ{i+1}",
            command=lambda v, l=leg, j=i: update(l,j,v)
        )
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s, leg, i))

# -----------------------------
# Manipulator UI
# -----------------------------
f = tk.LabelFrame(control, text="ARM")
f.pack(fill="x", pady=6)

for i in range(3):
    s = tk.Scale(
        f,
        from_=manip_limits[i][0],
        to=manip_limits[i][1],
        resolution=0.01,
        orient=tk.HORIZONTAL,
        label=f"θ{i+1}",
        command=lambda v, j=i: update_manip(j,v)
    )
    s.set(manip[i])
    s.pack(side="left", expand=True, fill="x")

    manip_sliders.append((s, i))

# -----------------------------
# Reset
# -----------------------------
def reset():
    for l in legs:
        legs[l] = defaults[l][:]
    for s, l, i in sliders:
        s.set(legs[l][i])

    for i in range(3):
        manip[i] = manip_default[i]
    for s, i in manip_sliders:
        s.set(manip[i])

    draw()
    canvas.draw()

tk.Button(control, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()