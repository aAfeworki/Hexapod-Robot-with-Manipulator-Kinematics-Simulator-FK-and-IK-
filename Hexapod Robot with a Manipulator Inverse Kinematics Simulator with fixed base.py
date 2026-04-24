import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# PARAMETERS
# -----------------------------
L1, L2 = 0.2, 0.2
Lm1, Lm2, Lm3 = 0.02, 0.2, 0.2

# IK targets
legs = {
    "FR": [0.0, 0.0, -0.3],
    "FL": [0.0, 0.0, -0.3],
    "MR": [0.0, 0.0, -0.3],
    "ML": [0.0, 0.0, -0.3],
    "RR": [0.0, 0.0, -0.3],
    "RL": [0.0, 0.0, -0.3]
}

arm_target = [0.0, 0.28, 0.2]
elbow_sign = 1

# Defaults
defaults_legs = {k: v[:] for k, v in legs.items()}
default_arm = arm_target[:]

# Limits
leg_limits = {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.6, -0.05)}
arm_limits = {"x": (-0.4, 0.5), "y": (-0.4, 0.4), "z": (0, 0.6)}

# -----------------------------
# HEXAGON BODY
# -----------------------------
hex_radius = 0.35
angles_body = np.deg2rad([0, 60, 120, 180, 240, 300])

body = np.array([[hex_radius * np.cos(a), hex_radius * np.sin(a), 0] for a in angles_body] +
                [[hex_radius * np.cos(angles_body[0]), hex_radius * np.sin(angles_body[0]), 0]])

base_pos = {
    "FR": body[5],
    "FL": body[4],
    "ML": body[3],
    "MR": body[0],
    "RL": body[2],
    "RR": body[1]
}

arm_base = np.array([0.0, 0.22, 0.0])


# -----------------------------
# IK FUNCTIONS
# -----------------------------
def ik_leg(x, y, z, flip_elbow=False):
    t1 = np.arctan2(y, -z)
    R = max(np.sqrt(y ** 2 + z ** 2), 1e-6)

    D = x ** 2 + R ** 2
    c = np.clip((D - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)

    # Standard t3 is Elbow Down. Flip to negative for Elbow Up.
    t3 = np.arccos(c)
    if flip_elbow:
        t3 = -t3

    t2 = np.arctan2(x, R) - np.arctan2(L2 * np.sin(t3), L1 + L2 * np.cos(t3))
    return t1, t2, t3


def ik_arm(X, Y, Z):
    global elbow_sign
    t1 = np.arctan2(Y, X)
    r = np.sqrt(X ** 2 + Y ** 2)
    z = Z - Lm1
    D = np.sqrt(r ** 2 + z ** 2)
    c = np.clip((D ** 2 - Lm2 ** 2 - Lm3 ** 2) / (2 * Lm2 * Lm3), -1, 1)
    t3 = elbow_sign * np.arccos(c)
    t2 = np.arctan2(z, r) - np.arctan2(Lm3 * np.sin(t3), Lm2 + Lm3 * np.cos(t3))
    return t1, t2, t3


# -----------------------------
# FK (for drawing)
# -----------------------------
def fk_leg(t1, t2, t3):
    X = L1 * np.sin(t2) + L2 * np.sin(t2 + t3)
    R = L1 * np.cos(t2) + L2 * np.cos(t2 + t3)
    return np.array([X, R * np.sin(t1), -R * np.cos(t1)])


def fk_arm(t1, t2, t3):
    X = (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.cos(t1)
    Y = (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.sin(t1)
    Z = Lm2 * np.sin(t2) + Lm3 * np.sin(t2 + t3) + Lm1
    return np.array([X, Y, Z])


# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


def draw():
    ax.clear()
    ax.plot(body[:, 0], body[:, 1], body[:, 2], linewidth=3)
    angles_dict = {}

    # ---- LEGS ----
    for leg, (x, y, z) in legs.items():
        base = base_pos[leg]

        # Right legs (FR, MR, RR) = Elbow Up
        is_right = leg in ["FR", "MR", "RR"]
        t1, t2, t3 = ik_leg(x, y, z, flip_elbow=is_right)

        angles_dict[leg] = np.degrees([t1, t2, t3])
        knee = base + [L1 * np.sin(t2), L1 * np.cos(t2) * np.sin(t1), -L1 * np.cos(t2) * np.cos(t1)]
        foot = base + fk_leg(t1, t2, t3)
        ax.plot(*zip(base, knee), marker='o')
        ax.plot(*zip(knee, foot), marker='o')

    # ---- ARM ----
    t1_a, t2_a, t3_a = ik_arm(*arm_target)
    angles_dict["ARM"] = np.degrees([t1_a, t2_a, t3_a])
    j1 = arm_base + [0, 0, Lm1]
    j2 = j1 + [Lm2 * np.cos(t2_a) * np.cos(t1_a), Lm2 * np.cos(t2_a) * np.sin(t1_a), Lm2 * np.sin(t2_a)]
    end = arm_base + fk_arm(t1_a, t2_a, t3_a)
    ax.plot(*zip(arm_base, j1), marker='o')
    ax.plot(*zip(j1, j2), marker='o')
    ax.plot(*zip(j2, end), marker='o')

    txt = "        θ1     θ2     θ3\n"
    for k in ["FR", "FL", "MR", "ML", "RR", "RL", "ARM"]:
        a = angles_dict[k]
        txt += f"{k}: {a[0]:6.1f} {a[1]:6.1f} {a[2]:6.1f}\n"

    ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9), va='top')
    ax.set(xlim=[-0.7, 0.7], ylim=[-0.7, 0.7], zlim=[-0.7, 0.7])


# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Hexapod IK + Arm Simulator")
root.geometry("1200x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel = tk.Frame(root)
panel.pack(side=tk.RIGHT, fill=tk.Y)

all_sliders = {}  # To keep track of sliders for reset


# -----------------------------
# LEG SLIDERS
# -----------------------------
def update_leg(l, i, v):
    legs[l][i] = float(v)
    draw();
    canvas.draw_idle()


for leg in legs:
    f = tk.LabelFrame(panel, text=leg)
    f.pack(fill="x", pady=3)
    all_sliders[leg] = []
    for i, n in enumerate(["X", "Y", "Z"]):
        s = tk.Scale(f, from_=leg_limits[n.lower()][0], to=leg_limits[n.lower()][1],
                     resolution=0.01, orient=tk.HORIZONTAL, label=n,
                     command=lambda v, l=leg, j=i: update_leg(l, j, v))
        s.set(legs[leg][i])
        s.pack(side="left", expand=True, fill="x")
        all_sliders[leg].append(s)


# -----------------------------
# ARM SLIDERS
# -----------------------------
def update_arm(i, v):
    arm_target[i] = float(v)
    draw();
    canvas.draw_idle()


f_arm = tk.LabelFrame(panel, text="ARM")
f_arm.pack(fill="x", pady=5)
all_sliders["ARM"] = []
for i, n in enumerate(["X", "Y", "Z"]):
    s = tk.Scale(f_arm, from_=arm_limits[n.lower()][0], to=arm_limits[n.lower()][1],
                 resolution=0.01, orient=tk.HORIZONTAL, label=n,
                 command=lambda v, j=i: update_arm(j, v))
    s.set(arm_target[i])
    s.pack(side="left", expand=True, fill="x")
    all_sliders["ARM"].append(s)

# -----------------------------
# BUTTONS
# -----------------------------
btn_frame = tk.Frame(panel)
btn_frame.pack(fill="x", pady=10)


def toggle_elbow():
    global elbow_sign
    elbow_sign *= -1
    draw();
    canvas.draw_idle()


def reset():
    global arm_target
    # Reset internal values
    for k in legs:
        legs[k] = defaults_legs[k][:]
        # Reset Sliders
        for i, val in enumerate(legs[k]):
            all_sliders[k][i].set(val)

    arm_target[:] = default_arm[:]
    for i, val in enumerate(arm_target):
        all_sliders["ARM"][i].set(val)

    draw();
    canvas.draw()


tk.Button(btn_frame, text="Elbow Up/Down", command=toggle_elbow).pack(side="left", expand=True, fill="x")
tk.Button(btn_frame, text="Reset", command=reset).pack(side="left", expand=True, fill="x")

draw()
canvas.draw()
root.mainloop()