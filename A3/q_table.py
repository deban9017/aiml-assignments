# I have a Q-table saved in 23607_disabled_1.pkl
# I want to print the Q-table in a readable format
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from tabulate import tabulate

# Load the Q-table
with open('23607_disabled_1.pkl', 'rb') as f:
    q_table = pickle.load(f)

table = q_table['best_q_table']
# # make the table (20,20, 4) into a (400,4) table
# table = table.reshape(-1, 4)
# # Print the Q-table
# print("Q-table:")
# print("--------------------------------------------------") 
# print(tabulate(table, headers='keys', tablefmt='rounded_grid'))

# # save the table to a file
# df = pd.DataFrame(table)
# df.to_csv('q_table.csv', index=False)


# the table is (20, 20, 4)
# we will peint 4 values separated by a comma within each cell for 20x20 table
# create a 20x20 table
table_2d = np.zeros((20, 20), dtype=object)
for i in range(20):
    for j in range(20):
        table_2d[i][j] = f"{table[i][j][0]}\n {table[i][j][1]}\n {table[i][j][2]}\n {table[i][j][3]}"
# Print the table
print("Q-table:")
print("--------------------------------------------------")
table_print = tabulate(table_2d, tablefmt='rounded_grid')
print(table_print)

# save the table to a file
with open('q_table.txt', 'w') as f:
    f.write(table_print)

