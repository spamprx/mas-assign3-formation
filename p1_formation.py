"""
Problem 1: Formation control to display the name "PRANEETH" letter-by-letter.

N = 20 single-integrator agents on an Erdos-Renyi random graph. Given desired
offsets r_i in R^2 for a letter, the distributed formation-control law
    u_i = sum_{j in N_i} a_ij [(p_j - p_i) - (r_j - r_i)]
drives p_i - p_j -> r_i - r_j (see MAS_Algorithms_Lecture, Algorithm 2).

We sequentially set the target offsets to letters P, R, A, N, E, E, T, H and
animate the whole trajectory using matplotlib.animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation

# ---------------- Parameters ----------------
N          = 20                 # number of agents
p_edge     = 0.30               # Erdos-Renyi edge probability
dt         = 0.05               # integration step
steps_per_letter = 400          # simulation steps per letter
seed       = 7
rng        = np.random.default_rng(seed)

NAME = "PRANEETH"

# ---------------- Erdos-Renyi connected graph ----------------
def connected_erdos_renyi(n, p, rng):
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1e9)))
        if nx.is_connected(G):
            return G

G = connected_erdos_renyi(N, p_edge, rng)
A = nx.to_numpy_array(G)                                 # adjacency (0/1)

# ---------------- Letter templates ----------------
# Each letter is described by strokes; a stroke is a pair of endpoints.
# We allocate the N agents across strokes proportionally to stroke length,
# so all parts of the letter get filled uniformly.
LETTERS = {
    "P": [((0, 0), (0, 3)), ((0, 3), (2, 3)), ((2, 3), (2, 1.5)), ((2, 1.5), (0, 1.5))],
    "R": [((0, 0), (0, 3)), ((0, 3), (2, 3)), ((2, 3), (2, 1.5)),
          ((2, 1.5), (0, 1.5)), ((0.7, 1.5), (2, 0))],
    "A": [((0, 0), (1, 3)), ((1, 3), (2, 0)), ((0.5, 1.2), (1.5, 1.2))],
    "N": [((0, 0), (0, 3)), ((0, 3), (2, 0)), ((2, 0), (2, 3))],
    "E": [((0, 0), (0, 3)), ((0, 3), (2, 3)), ((0, 1.5), (1.5, 1.5)), ((0, 0), (2, 0))],
    "T": [((0, 3), (2, 3)), ((1, 3), (1, 0))],
    "H": [((0, 0), (0, 3)), ((2, 0), (2, 3)), ((0, 1.5), (2, 1.5))],
}


def letter_positions(ch, n):
    """Return n points that fill the strokes of the letter, ordered along the letter."""
    strokes = LETTERS[ch]
    lengths = np.array([np.linalg.norm(np.array(b) - np.array(a)) for a, b in strokes])
    total = lengths.sum()
    counts = np.maximum(2, np.round(n * lengths / total).astype(int))
    # adjust counts so they sum to exactly n
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n:
        counts[np.argmin(counts)] += 1
    pts = []
    for (a, b), c in zip(strokes, counts):
        a = np.array(a, float); b = np.array(b, float)
        for k in range(c):
            t = k / max(1, c - 1)
            pts.append(a + t * (b - a))
    return np.array(pts[:n])


# ---------------- Simulation ----------------
# Random initial positions in a 6x6 box
p = rng.uniform(-3.0, 3.0, size=(N, 2))
trajectory = [p.copy()]

L = np.diag(A.sum(axis=1)) - A   # graph Laplacian

# Pure formation control leaves the centroid invariant; to place each letter
# nicely on screen we also add a small centroid-tracking term  bi*(c - p_i)
# with a single pinning agent (agent 0) that knows the desired centroid c=0.
b_pin = np.zeros(N); b_pin[0] = 1.0
c_target = np.zeros(2)

for ch in NAME:
    r_raw = letter_positions(ch, N)
    r = r_raw - r_raw.mean(axis=0)          # center letter around origin
    for _ in range(steps_per_letter):
        # Formation control (Algorithm 2 of MAS_Algorithms_Lecture)
        u = -L @ (p - r)
        # add mild leader-pinning so the whole formation stays around c_target
        u += b_pin[:, None] * (c_target + r - p)
        p = p + dt * u
        trajectory.append(p.copy())

traj = np.array(trajectory)                 # shape: (T, N, 2)
print(f"Simulation done: {traj.shape[0]} frames, N={N}, graph edges={G.number_of_edges()}")

# ---------------- Static snapshots ----------------
fig, axs = plt.subplots(1, len(NAME), figsize=(2.2 * len(NAME), 3), sharey=True)
for k, ch in enumerate(NAME):
    frame_idx = (k + 1) * steps_per_letter    # end of that letter
    pts = traj[frame_idx]
    axs[k].scatter(pts[:, 0], pts[:, 1], s=25, c="steelblue")
    axs[k].set_title(ch)
    axs[k].set_aspect("equal")
    axs[k].set_xlim(-1.5, 1.5)
    axs[k].set_ylim(-2, 2)
    axs[k].grid(alpha=0.3)
plt.suptitle("Final formation for each letter")
plt.tight_layout()
plt.savefig("p1_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------- Animation ----------------
fig2, ax2 = plt.subplots(figsize=(5, 5))
sc = ax2.scatter(traj[0, :, 0], traj[0, :, 1], s=30, c="crimson")

# draw graph edges (light gray)
edge_lines = []
for i, j in G.edges():
    ln, = ax2.plot([traj[0, i, 0], traj[0, j, 0]],
                   [traj[0, i, 1], traj[0, j, 1]],
                   color="lightgray", lw=0.6, zorder=1)
    edge_lines.append((i, j, ln))

ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect("equal")
ax2.grid(alpha=0.3)
title = ax2.set_title("")


def update(k):
    pts = traj[k]
    sc.set_offsets(pts)
    for i, j, ln in edge_lines:
        ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
    letter_idx = min(k // steps_per_letter, len(NAME) - 1)
    title.set_text(f"Forming letter '{NAME[letter_idx]}' (frame {k})")
    return sc,


ani = animation.FuncAnimation(fig2, update, frames=range(0, traj.shape[0], 3),
                              interval=30, blit=False)

# Save as mp4 if ffmpeg is available, otherwise as gif
try:
    ani.save("p1_formation.mp4", fps=30, dpi=150)
    print("Saved p1_formation.mp4")
except Exception as e:
    ani.save("p1_formation.gif", fps=30, dpi=100, writer="pillow")
    print("Saved p1_formation.gif")

# Also save the graph plot
figG, axG = plt.subplots(figsize=(4, 4))
nx.draw(G, node_color="steelblue", edge_color="gray", node_size=180,
        with_labels=True, ax=axG)
axG.set_title(f"Erdos-Renyi graph (N={N}, p={p_edge})")
plt.tight_layout()
plt.savefig("p1_graph.png", dpi=150, bbox_inches="tight")
plt.close()
