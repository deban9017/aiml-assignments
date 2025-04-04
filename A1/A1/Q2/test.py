# import sys
# import os
# import numpy as np


# # Add the 'oracle' directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
# import oracle as oracle 

# def main():
#     res = oracle.q2_train_test_emnist(23607, "./EMNIST/emnist-balanced-train.csv", "./EMNIST/emnist-balanced-test.csv")
#     with open("q2_results.txt", "w") as f:
#         f.write(str(res))

# if __name__ == '__main__':
#     main()

import matplotlib.pyplot as plt

# Data for the first table (test split 50-50)
ratios1 = ['40/60', '20/80', '10/90', '1/99']
loss1 = [0.026, 0.484, 0.500, 0.500]

# Data for the second table (same split in both train and test)
ratios2 = ['40/60', '20/80', '10/90', '1/99']
loss2 = [0.030, 0.198, 0.099, 0.010]

# Convert ratios to numerical values for x-axis (optional, for better plotting order)
def ratio_to_float(ratio_str):
    num, den = map(int, ratio_str.split('/'))
    return num / den

x_values = [ratio_to_float(r) for r in ratios1] # Using ratios1 as ratios are same for both

# Sort the data based on x_values to ensure correct order in plot if needed
data1 = sorted(zip(x_values, ratios1, loss1), key=lambda x: x[0])
x_values_sorted = [item[0] for item in data1]
ratios1_sorted = [item[1] for item in data1]
loss1_sorted = [item[2] for item in data1]

data2 = sorted(zip(x_values, ratios2, loss2), key=lambda x: x[0])
ratios2_sorted = [item[1] for item in data2]
loss2_sorted = [item[2] for item in data2]


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ratios1_sorted, loss1_sorted, marker='o', linestyle='-', color='blue', label='Test split 50-50')
plt.plot(ratios2_sorted, loss2_sorted, marker='o', linestyle='-', color='red', label='Same split train/test')

plt.xlabel('Train/Test Split Ratio (Train:Test)')
plt.ylabel('Misclassification Loss')
plt.title('Misclassification Loss vs. Train/Test Split Ratio')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.grid(True)
plt.legend()
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()