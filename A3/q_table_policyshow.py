import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

enabled = "disabled"
num = 2
# Load Q-table
with open(f'23607_{enabled}_{num}.pkl', 'rb') as f:
    q_table = pickle.load(f)

table = q_table['best_q_table']  # shape (20, 20, 4)

grid_size = 20
action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
arrow_grid = np.full((grid_size, grid_size), '', dtype=object)  # Initialize empty

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')

# Gridlines at cell borders
ax.set_xticks(np.arange(0, grid_size + 1), minor=True)
ax.set_yticks(np.arange(0, grid_size + 1), minor=True)
ax.grid(which='minor', color='black', linewidth=0.5)

# Hide major ticks
ax.set_xticks([])
ax.set_yticks([])

# Fill grid with arrows or black cells
for i in range(grid_size):
    for j in range(grid_size):
        q_values = table[i, j]
        if np.all(q_values == 0):
            # Draw black rectangle for unvisited cell
            rect = plt.Rectangle((j, grid_size - i - 1), 1, 1, color='black')
            ax.add_patch(rect)
        else:
            action = np.argmax(q_values)
            arrow = action_arrows[action]
            ax.text(j + 0.5, grid_size - i - 0.5, arrow, ha='center', va='center', fontsize=14)

ax.set_title("Q-Table Policy: Arrows + Unvisited Cells (Black)", fontsize=16)
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig(f'policy_{enabled}_{num}.png', dpi=300)