import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tabulate
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
import dtreeviz


# Add the 'oracle' directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'oracle'))
import oracle

###############################################################################
###############################################################################
#  ███████╗██╗     ██████╗ 
#  ██╔════╝██║     ██╔══██╗
#  █████╗  ██║     ██║  ██║
#  ██╔══╝  ██║     ██║  ██║
#  ██║     ███████╗██████╔╝
#  ╚═╝     ╚══════╝╚═════╝ 
###############################################################################
###############################################################################

res = oracle.q1_fish_train_test_data(23607)
print(res[0])

attributes = res[0]
train_img = np.array(res[1])
train_labels = np.array(res[2])
test_img = np.array(res[3])
test_labels = np.array(res[4])

# flatten the images
train_img = train_img.reshape(train_img.shape[0], -1)
test_img = test_img.reshape(test_img.shape[0], -1)

train_img_0 = train_img[train_labels == 0]
train_img_1 = train_img[train_labels == 1]
train_img_2 = train_img[train_labels == 2]
train_img_3 = train_img[train_labels == 3]

test_img_0 = test_img[test_labels == 0]
test_img_1 = test_img[test_labels == 1]
test_img_2 = test_img[test_labels == 2]
test_img_3 = test_img[test_labels == 3]

c = 4 # number of classes

# _____________________________________________________________________________

def fld_w(train_img, n):
    W_T = []

    train_img_0_n = train_img[0][:n]
    train_img_1_n = train_img[1][:n]
    train_img_2_n = train_img[2][:n]
    train_img_3_n = train_img[3][:n]

    m0_n = np.mean(train_img_0_n, axis=0)
    m1_n = np.mean(train_img_1_n, axis=0)
    m2_n = np.mean(train_img_2_n, axis=0)
    m3_n = np.mean(train_img_3_n, axis=0)

    # Scatter matrix formula: S = ∑ (x - m) (x - m)^T
    s0_n = np.cov(train_img_0_n.T)*n
    s1_n = np.cov(train_img_1_n.T)*n
    s2_n = np.cov(train_img_2_n.T)*n
    s3_n = np.cov(train_img_3_n.T)*n

    s_W_n = s0_n + s1_n + s2_n + s3_n

    means = np.array([m0_n, m1_n, m2_n, m3_n])
    m = np.mean(means, axis=0)
    s_B_n = np.zeros((train_img_0_n.shape[1], train_img_0_n.shape[1]))
    for i in range(c):
        diff = (means[i] - m).reshape(-1, 1)
        s_B_n += n * np.dot(diff, diff.T)
    
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(s_W_n).dot(s_B_n))
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    W_T.append(eigvecs[:, :c-1]) # We take the first c-1 eigenvectors
    return W_T
# _____________________________________________________________________________
def accuracy(test_img, test_labels, w, train_img_n, n):
    w = fld_w(train_img_n, n)

    train_img_0_proj = train_img_n[0].dot(w[0])
    train_img_1_proj = train_img_n[1].dot(w[0])
    train_img_2_proj = train_img_n[2].dot(w[0])
    train_img_3_proj = train_img_n[3].dot(w[0])

    m0_proj = np.mean(train_img_0_proj, axis=0)
    m1_proj = np.mean(train_img_1_proj, axis=0)
    m2_proj = np.mean(train_img_2_proj, axis=0)
    m3_proj = np.mean(train_img_3_proj, axis=0)

    cov0_proj = np.cov(train_img_0_proj.T)
    cov1_proj = np.cov(train_img_1_proj.T)
    cov2_proj = np.cov(train_img_2_proj.T)
    cov3_proj = np.cov(train_img_3_proj.T)

    false_count = 0
    class_pred = []

    for x in test_img:
        x_proj = x.dot(w[0])
        posteriors = [
            np.exp(-0.5 * np.dot(np.dot((x_proj - m0_proj), np.linalg.inv(cov0_proj)), (x_proj - m0_proj).T)),
            np.exp(-0.5 * np.dot(np.dot((x_proj - m1_proj), np.linalg.inv(cov1_proj)), (x_proj - m1_proj).T)),
            np.exp(-0.5 * np.dot(np.dot((x_proj - m2_proj), np.linalg.inv(cov2_proj)), (x_proj - m2_proj).T)),
            np.exp(-0.5 * np.dot(np.dot((x_proj - m3_proj), np.linalg.inv(cov3_proj)), (x_proj - m3_proj).T))
        ]
        class_pred.append(np.argmax(posteriors))

    class_pred = np.array(class_pred)
    false_count = np.sum(class_pred != test_labels)  


    return false_count/len(test_labels)
# _____________________________________________________________________________
n_all = [200, 500, 1000, 1500, 2000, 2500, 3500, 4000, 4500, 5000]
accuracy_values = []
for n in n_all:
    #sample n images
    train_0 = train_img_0[np.random.choice(train_img_0.shape[0], n, replace=False)]
    train_1 = train_img_1[np.random.choice(train_img_1.shape[0], n, replace=False)]
    train_2 = train_img_2[np.random.choice(train_img_2.shape[0], n, replace=False)]
    train_3 = train_img_3[np.random.choice(train_img_3.shape[0], n, replace=False)]
    train_img_n = [train_0, train_1, train_2, train_3]

    accuracy_values.append(accuracy(test_img, test_labels, fld_w(train_img_n, n), train_img_n, n))

plt.figure(figsize=(8, 6))
plt.plot(n_all, accuracy_values, marker='o', linestyle='-')

plt.xlabel('n')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for different values of n')
plt.xticks(n_all)
plt.tight_layout()
plt.show()

# _____________________________________________________________________________
# Mean of each class
m0 = np.mean(train_img_0, axis=0)
m1 = np.mean(train_img_1, axis=0)
m2 = np.mean(train_img_2, axis=0)
m3 = np.mean(train_img_3, axis=0)

# Covariance matrix of each class
s1 = np.cov(train_img_0.T)
s2 = np.cov(train_img_1.T)
s3 = np.cov(train_img_2.T)
s4 = np.cov(train_img_3.T)

# total mean
m_T = (m0 + m1 + m2 + m3) / 4

print("Total mean norm",np.linalg.norm(m_T))
print("c0 mean norm",np.linalg.norm(m0))
print("c1 mean norm",np.linalg.norm(m1))
print("c2 mean norm",np.linalg.norm(m2))
print("c3 mean norm",np.linalg.norm(m3))

# n = 50
train_img_0_50 = train_img_0[:50]
train_img_1_50 = train_img_1[:50]
train_img_2_50 = train_img_2[:50]
train_img_3_50 = train_img_3[:50]

# n = 100
train_img_0_100 = train_img_0[:100]
train_img_1_100 = train_img_1[:100]
train_img_2_100 = train_img_2[:100]
train_img_3_100 = train_img_3[:100]

# n = 500
train_img_0_500 = train_img_0[:500]
train_img_1_500 = train_img_1[:500]
train_img_2_500 = train_img_2[:500]
train_img_3_500 = train_img_3[:500]

# n = 1000
train_img_0_1000 = train_img_0[:1000]
train_img_1_1000 = train_img_1[:1000]
train_img_2_1000 = train_img_2[:1000]
train_img_3_1000 = train_img_3[:1000]

# n = 2000
train_img_0_2000 = train_img_0[:2000]
train_img_1_2000 = train_img_1[:2000]
train_img_2_2000 = train_img_2[:2000]
train_img_3_2000 = train_img_3[:2000]

# n = 5000
train_img_0_5000 = train_img_0[:5000]
train_img_1_5000 = train_img_1[:5000]
train_img_2_5000 = train_img_2[:5000]
train_img_3_5000 = train_img_3[:5000]

# headers c0, c1, c2, c3; rows n=50, n=100, n=200, n=500, n=1000, n=2000, n=5000
table = [
    ["n=50", np.linalg.norm(np.mean(train_img_0_50, axis=0)), np.linalg.norm(np.mean(train_img_1_50, axis=0)), np.linalg.norm(np.mean(train_img_2_50, axis=0)), np.linalg.norm(np.mean(train_img_3_50, axis=0))],
    ["n=100", np.linalg.norm(np.mean(train_img_0_100, axis=0)), np.linalg.norm(np.mean(train_img_1_100, axis=0)), np.linalg.norm(np.mean(train_img_2_100, axis=0)), np.linalg.norm(np.mean(train_img_3_100, axis=0))],
    ["n=500", np.linalg.norm(np.mean(train_img_0_500, axis=0)), np.linalg.norm(np.mean(train_img_1_500, axis=0)), np.linalg.norm(np.mean(train_img_2_500, axis=0)), np.linalg.norm(np.mean(train_img_3_500, axis=0))],
    ["n=1000", np.linalg.norm(np.mean(train_img_0_1000, axis=0)), np.linalg.norm(np.mean(train_img_1_1000, axis=0)), np.linalg.norm(np.mean(train_img_2_1000, axis=0)), np.linalg.norm(np.mean(train_img_3_1000, axis=0))],
    ["n=2000", np.linalg.norm(np.mean(train_img_0_2000, axis=0)), np.linalg.norm(np.mean(train_img_1_2000, axis=0)), np.linalg.norm(np.mean(train_img_2_2000, axis=0)), np.linalg.norm(np.mean(train_img_3_2000, axis=0))],
    ["n=5000", np.linalg.norm(np.mean(train_img_0_5000, axis=0)), np.linalg.norm(np.mean(train_img_1_5000, axis=0)), np.linalg.norm(np.mean(train_img_2_5000, axis=0)), np.linalg.norm(np.mean(train_img_3_5000, axis=0))],
]

print("Mean norms of each class for different n:")
print(tabulate.tabulate(table, headers=["c0", "c1", "c2", "c3"], tablefmt="rounded_outline"))

# Plot
n_values = ["n=50", "n=100", "n=500", "n=1000", "n=2000", "n=4000"]
c0_values = [row[1] for row in table]
c1_values = [row[2] for row in table]
c2_values = [row[3] for row in table]
c3_values = [row[4] for row in table]

plt.figure(figsize=(10, 6))
plt.plot(n_values, c0_values, marker='o', label='c0')
plt.plot(n_values, c1_values, marker='o', label='c1')
plt.plot(n_values, c2_values, marker='o', label='c2')
plt.plot(n_values, c3_values, marker='o', label='c3')

plt.xlabel('Classes')
plt.ylabel('Mean Norms')
plt.title('Mean Norms of Each Class for Different n')
plt.legend()
plt.grid(True)
plt.show()
# _____________________________________________________________________________
# NOTE: np.linalg.norm() automatically take frobenius norm for matrices.
table = [
    ["n=50", np.linalg.norm(np.cov(train_img_0_50.T)), np.linalg.norm(np.cov(train_img_1_50.T)), np.linalg.norm(np.cov(train_img_2_50.T)), np.linalg.norm(np.cov(train_img_3_50.T))],
    ["n=100", np.linalg.norm(np.cov(train_img_0_100.T)), np.linalg.norm(np.cov(train_img_1_100.T)), np.linalg.norm(np.cov(train_img_2_100.T)), np.linalg.norm(np.cov(train_img_3_100.T))],
    ["n=500", np.linalg.norm(np.cov(train_img_0_500.T)), np.linalg.norm(np.cov(train_img_1_500.T)), np.linalg.norm(np.cov(train_img_2_500.T)), np.linalg.norm(np.cov(train_img_3_500.T))],
    ["n=1000", np.linalg.norm(np.cov(train_img_0_1000.T)), np.linalg.norm(np.cov(train_img_1_1000.T)), np.linalg.norm(np.cov(train_img_2_1000.T)), np.linalg.norm(np.cov(train_img_3_1000.T))],
    ["n=2000", np.linalg.norm(np.cov(train_img_0_2000.T)), np.linalg.norm(np.cov(train_img_1_2000.T)), np.linalg.norm(np.cov(train_img_2_2000.T)), np.linalg.norm(np.cov(train_img_3_2000.T))],
    ["n=4000", np.linalg.norm(np.cov(train_img_0_5000.T)), np.linalg.norm(np.cov(train_img_1_5000.T)), np.linalg.norm(np.cov(train_img_2_5000.T)), np.linalg.norm(np.cov(train_img_3_5000.T))],
]

print("Covariance norms of each class for different n:")
print(tabulate.tabulate(table, headers=["c0", "c1", "c2", "c3"], tablefmt="rounded_outline"))

# Plotting the graph
n_values = ["n=50", "n=100", "n=500", "n=1000", "n=2000", "n=4000"]
c0_values = [row[1] for row in table]
c1_values = [row[2] for row in table]
c2_values = [row[3] for row in table]
c3_values = [row[4] for row in table]

plt.figure(figsize=(10, 6))
plt.plot(n_values, c0_values, marker='o', label='c0')
plt.plot(n_values, c1_values, marker='o', label='c1')
plt.plot(n_values, c2_values, marker='o', label='c2')
plt.plot(n_values, c3_values, marker='o', label='c3')

plt.xlabel('Classes')
plt.ylabel('Cov Norms')
plt.title('Cov Norms of Each Class for Different n')
plt.legend()
plt.grid(True)
plt.show()
# _____________________________________________________________________________
W = fld_w(train_img_n, 5000)
test_img_proj = test_img.dot(W)
train_img_proj = train_img.dot(W)
#_____________________________________________________________________________
# Plotting the 3D scatter plot of the projected train images
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_img_proj[train_labels == 0, 0], train_img_proj[train_labels == 0, 1], train_img_proj[train_labels == 0, 2], label='c0')
ax.scatter(train_img_proj[train_labels == 1, 0], train_img_proj[train_labels == 1, 1], train_img_proj[train_labels == 1, 2], label='c1')
ax.scatter(train_img_proj[train_labels == 2, 0], train_img_proj[train_labels == 2, 1], train_img_proj[train_labels == 2, 2], label='c2')
ax.scatter(train_img_proj[train_labels == 3, 0], train_img_proj[train_labels == 3, 1], train_img_proj[train_labels == 3, 2], label='c3')
plt.title('Projected Train Images in 3D')
plt.legend()
plt.show()
#_____________________________________________________________________________
# Plotting the 3D scatter plot of the projected test images
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_img_proj[test_labels == 0, 0], test_img_proj[test_labels == 0, 1], test_img_proj[test_labels == 0, 2], label='c0')
ax.scatter(test_img_proj[test_labels == 1, 0], test_img_proj[test_labels == 1, 1], test_img_proj[test_labels == 1, 2], label='c1')
ax.scatter(test_img_proj[test_labels == 2, 0], test_img_proj[test_labels == 2, 1], test_img_proj[test_labels == 2, 2], label='c2')
ax.scatter(test_img_proj[test_labels == 3, 0], test_img_proj[test_labels == 3, 1], test_img_proj[test_labels == 3, 2], label='c3')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('Projected Test Images in 3D')
plt.legend()
plt.show()
#_____________________________________________________________________________
# Plotting the objective values, has redundancy, function created at last moment, could not use that.
n_all = [2500, 3500, 4000, 4500,5000] 
W_T_all = []
c = 4 # number of classes

objective_values = {} # key: n, value: 20 objective values list

for n in n_all:
    print("n:", n, "__________________________")
    rng = 20
    if n == 5000:
        rng = 1

    for i in range(rng):
        print("Iteration:", i)

        indices = np.random.choice(train_img_0.shape[0], n, replace=False)
        train_img_0_n = train_img_0[indices]
        train_img_1_n = train_img_1[indices]
        train_img_2_n = train_img_2[indices]
        train_img_3_n = train_img_3[indices]

        m0_n = np.mean(train_img_0_n, axis=0)
        m1_n = np.mean(train_img_1_n, axis=0)
        m2_n = np.mean(train_img_2_n, axis=0)
        m3_n = np.mean(train_img_3_n, axis=0)

        # Calculate scatter matrices using the proper formula: S = ∑ (x - m) (x - m)^T
        s0_n = np.cov(train_img_0_n.T)*(n-1)
        s1_n = np.cov(train_img_1_n.T)*(n-1)
        s2_n = np.cov(train_img_2_n.T)*(n-1)
        s3_n = np.cov(train_img_3_n.T)*(n-1)

        print("Calculating s_W and s_B")
        s_W_n = s0_n + s1_n + s2_n + s3_n
        print("s_W_n shape:", s_W_n.shape)
        m = (m0_n + m1_n + m2_n + m3_n) / 4  # total mean
        s_B_n = n * ((m0_n - m).reshape(-1, 1).dot((m0_n - m).reshape(-1, 1).T) + (m1_n - m).reshape(-1, 1).dot((m1_n - m).reshape(-1, 1).T) + (m2_n - m).reshape(-1, 1).dot((m2_n - m).reshape(-1, 1).T) + (m3_n - m).reshape(-1, 1).dot((m3_n - m).reshape(-1, 1).T))

        print("Calculating eigenvalues")
        # We calculate the eigenvalues and eigenvectors of s_W^-1 * s_B
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(s_W_n).dot(s_B_n))
        eigvals = eigvals.real

        eigvals_sorted = np.sort(eigvals)[::-1]

        # Sum the first three eigenvalues
        sum_top3_eigvals = np.sum(eigvals_sorted[:3])
        print("Sum of top 3 eigenvalues:", sum_top3_eigvals)

        objective_values[n] = objective_values.get(n, [])
        objective_values[n].append(sum_top3_eigvals)

    print(objective_values)
# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot(objective_values.values(), labels=objective_values.keys())
plt.xlabel('n')
plt.ylabel('Objective Value')
plt.title('Objective Values for Different n')
plt.show()
#_____________________________________________________________________________
#_____________________________________________________________________________

###############################################################################
###############################################################################
#  ██████╗   █████╗ ██╗   ██╗███████╗███████╗
#  ██╔══██╗ ██╔══██╗╚██╗ ██╔╝██╔════╝██╔════╝
#  ██████╔╝ ███████║ ╚████╔╝ █████╗  ███████╗
#  ██╔══██╗ ██╔══██║  ╚██╔╝  ██╔══╝  ╚════██║
#  ██████╔╝ ██║  ██║   ██║   ███████╗███████║
#  ╚═════╝  ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝
###############################################################################
###############################################################################


res = oracle.q2_train_test_emnist(23607, "./EMNIST/emnist-balanced-train.csv", "./EMNIST/emnist-balanced-test.csv")
train = res[0]
test = res[1]

class_count = {}
for i in range(len(train)):
    label = train[i][0]
    if label not in class_count:
        class_count[label] = 1
    else:
        class_count[label] += 1
print(class_count)

class_1 = 5
class_2 = 27

def h(x, m1, m2):
    return np.dot((m1 - m2).T , (x - (m1 + m2)/2))

m1 = np.zeros(784)
m2 = np.zeros(784)

for i in range(len(train)):
    if train[i][0] == class_1:
        m1 += np.array(train[i][1:])
    else:
        m2 += np.array(train[i][1:])
m1 /= class_count[class_1]
m2 /= class_count[class_2]

def result(train, test, epsilon):
    d = 784
    train_class5 = []
    train_class27 = []
    class_count = {class_1: 0, class_2: 0}
    for i in range(len(train)):
        if train[i][0] == class_1:
            train_class5.append(train[i][1:])
            class_count[class_1] += 1
        else:
            train_class27.append(train[i][1:])
            class_count[class_2] += 1
    
    train_class5 = np.array(train_class5, dtype=np.float64)
    train_class27 = np.array(train_class27, dtype=np.float64)

    m1 = np.mean(train_class5, axis=0)
    m2 = np.mean(train_class27, axis=0)

    cov5 = np.cov(train_class5.T)
    cov27 = np.cov(train_class27.T)

    # Regularization: add a small constant to the diagonal elements //Took suggestion from GPT
    eps = 1e-6 
    cov5_reg = cov5 + eps * np.eye(cov5.shape[0])
    cov27_reg = cov27 + eps * np.eye(cov27.shape[0])

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    no_class_count = 0  
    for i,data in enumerate(test):
        eta5 = log_eta(class_count[class_1] / len(train), log_N(data[1:], m1, cov5_reg, d))
        eta27 = log_eta(class_count[class_2] / len(train), log_N(data[1:], m2, cov27_reg, d))

        max_log = max(eta5, eta27)
        exp_eta5 = np.exp(eta5 - max_log)
        exp_eta27 = np.exp(eta27 - max_log)

        posterior_5 = exp_eta5 / (exp_eta5 + exp_eta27)
        posterior_27 = exp_eta27 / (exp_eta5 + exp_eta27)

        if abs(posterior_5 - posterior_27) < 2*epsilon:
            no_class_count += 1
            continue
        if posterior_5 < posterior_27 and data[0] == class_1:
            FN += 1
        elif posterior_5 > posterior_27 and data[0] == class_2:
            FP += 1
        elif posterior_5 > posterior_27 and data[0] == class_1:
            TP += 1
        elif posterior_5 < posterior_27 and data[0] == class_2:
            TN += 1
    return {'no_class_count': no_class_count, 'class5_count': class_count[class_1], 'class27_count': class_count[class_2], 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def log_N(x, m, cov, d):
    """
    Compute the log of the multivariate normal density for a given x.
    """
    from math import log, pi
    # Using slogdet for better numerical stability //Took suggestion from GPT
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    const = -d / 2 * log(2 * pi) - 0.5 * logdet
    # Here, using np.linalg.inv is acceptable because we've regularized the matrix. //Took suggestion from GPT
    exp_term = -0.5 * np.dot(np.dot((x - m).T, np.linalg.inv(cov)), (x - m))
    return const + exp_term

def log_eta(p, log_N_):
    """
    Combine the log prior p and the log likelihood to compute a log-posterior (up to a constant).
    """
    from math import log
    return log(p) + log_N_

#_____________________________________________________________________________
ratios = [1, 40/60, 20/80, 10/90, 1/99]
for ratio in ratios:
    print("ratio",ratio)
    res = oracle.q2_train_test_emnist(23607, "./EMNIST/emnist-balanced-train.csv", "./EMNIST/emnist-balanced-test.csv")
    train = res[0]
    test = res[1]
    train_class5 = []
    train_class27 = []
    for i in range(len(train)):
        if train[i][0] == class_1:
            train_class5.append(train[i])
        else:
            train_class27.append(train[i])
    test_class5 = []
    test_class27 = []
    for i in range(len(test)):
        if test[i][0] == class_1:
            test_class5.append(test[i])
        else:
            test_class27.append(test[i])   

    train_class5 = train_class5[:]
    train_class27 = train_class27[:int(ratio*len(train_class27))]
    train = train_class5 + train_class27

    test_class5 = test_class5[:]
    test_class27 = test_class27[:int(ratio*len(test_class27))]
    test = test_class5 + test_class27

    res = result(train, test, 0.4)
    print(res)
    print("______________")
"""
EXAMPLE OUTPUT:
_____________________________________
ratio 0.6666666666666666
{'no_class_count': 0, 'class5_count': 2400, 'class27_count': 1600, 'TP': 384, 'TN': 262, 'FP': 4, 'FN': 16}
______________
ratio 0.25
{'no_class_count': 0, 'class5_count': 2400, 'class27_count': 600, 'TP': 400, 'TN': 1, 'FP': 99, 'FN': 0}
______________
ratio 0.1111111111111111
{'no_class_count': 0, 'class5_count': 2400, 'class27_count': 266, 'TP': 400, 'TN': 0, 'FP': 44, 'FN': 0}
______________
ratio 0.010101010101010102
{'no_class_count': 0, 'class5_count': 2400, 'class27_count': 24, 'TP': 400, 'TN': 0, 'FP': 4, 'FN': 0}
______________
"""
#_____________________________________________________________________________
#_____________________________________________________________________________
# K FOLD CROSS VALIDATION, K = 5

def train_model(train):
    """
    Train the Modified Bayes classifier.
    Returns a dictionary of model parameters.
    """
    d = 784
    train_class5 = []
    train_class27 = []
    class_count = {class_1: 0, class_2: 0}
    for sample in train:
        if sample[0] == class_1:
            train_class5.append(sample[1:])
            class_count[class_1] += 1
        else:
            train_class27.append(sample[1:])
            class_count[class_2] += 1

    train_class5 = np.array(train_class5, dtype=np.float64)
    train_class27 = np.array(train_class27, dtype=np.float64)

    m1 = np.mean(train_class5, axis=0)
    m2 = np.mean(train_class27, axis=0)

    cov5 = np.cov(train_class5.T)
    cov27 = np.cov(train_class27.T)

    # Regularization for covariance matrices //Took suggestion from GPT
    reg_eps = 1e-6
    cov5_reg = cov5 + reg_eps * np.eye(cov5.shape[0])
    cov27_reg = cov27 + reg_eps * np.eye(cov27.shape[0])

    model = {
        'm1': m1,
        'm2': m2,
        'cov5_reg': cov5_reg,
        'cov27_reg': cov27_reg,
        'class_count': class_count,
        'd': d,
        'total_train': len(train)
    }
    return model

def evaluate_model(model, dataset, epsilon):
    """
    Evaluate the model on a dataset.
    Returns confusion matrix counts and other statistics.
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    no_class_count = 0 

    for sample in dataset:
        # sample[0] is true label; sample[1:] are features
        eta5 = log_eta(model['class_count'][class_1] / model['total_train'], 
                       log_N(sample[1:], model['m1'], model['cov5_reg'], model['d']))
        eta27 = log_eta(model['class_count'][class_2] / model['total_train'], 
                        log_N(sample[1:], model['m2'], model['cov27_reg'], model['d']))
        max_log = max(eta5, eta27)
        exp_eta5 = np.exp(eta5 - max_log)
        exp_eta27 = np.exp(eta27 - max_log)
        posterior_5 = exp_eta5 / (exp_eta5 + exp_eta27)
        posterior_27 = exp_eta27 / (exp_eta5 + exp_eta27)
        
        if abs(posterior_5 - posterior_27) < 2 * epsilon:
            no_class_count += 1
            continue
        
        if posterior_5 > posterior_27:
            if sample[0] == class_1:
                TP += 1
            else:
                FP += 1
        else:
            if sample[0] == class_2:
                TN += 1
            else:
                FN += 1

    non_rejected = TP + TN + FP + FN
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'no_class_count': no_class_count,
        'non_rejected': non_rejected,
        'total': len(dataset)
    }

#Cross-Validation and Best Model Selection_____________________________________
# agin loading, since some problem was occuring
res = oracle.q2_train_test_emnist(23607, "./EMNIST/emnist-balanced-train.csv", "./EMNIST/emnist-balanced-test.csv")
train_full = res[0]  # full training data
test_data = res[1]

# Assume class_1 and class_2 are defined (e.g., class_1 = 5, class_2 = 27)
# Separate training data by class
train_class5 = [sample for sample in train_full if sample[0] == class_1]
train_class27 = [sample for sample in train_full if sample[0] == class_2]

# Since classes are balanced, l is the number of samples in one class.
l = len(train_class5)
k = 5
n = l // k

# Shuffle each class separately
np.random.shuffle(train_class5)
np.random.shuffle(train_class27)

# Build k folds (each fold has equal samples from both classes)
folds = []
for i in range(k):
    fold = train_class5[i*n:(i+1)*n] + train_class27[i*n:(i+1)*n]
    np.random.shuffle(fold)
    folds.append(fold)

# Cross-validation loop: train on k-1 folds, validate on the remaining fold.
best_model = None
best_accuracy = -1
cv_accuracies = []

for i in range(k):
    # Build training and validation sets for fold i
    cv_train = []
    for j in range(k):
        if j != i:
            cv_train.extend(folds[j])
    cv_val = folds[i]
    
    # Train model on cv_train
    model = train_model(cv_train)
    # Evaluate on validation set (using epsilon=0.25 for the reject threshold)
    metrics = evaluate_model(model, cv_val, epsilon=0.25)
    print(f"Fold {i+1}:___________ ")
    print(metrics)
    # Compute accuracy on non-rejected samples
    if metrics['non_rejected'] > 0:
        accuracy = (metrics['TP'] + metrics['TN']) / metrics['non_rejected']
    else:
        accuracy = 0
    
    cv_accuracies.append(accuracy)
    print(f"Fold {i+1}: Accuracy = {accuracy:.4f}, Rejection Rate = {metrics['no_class_count'] / metrics['total']:.4f}")
    
    # Choose the best model (highest accuracy on non-rejected samples)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Average CV Accuracy: {:.4f}".format(np.mean(cv_accuracies)))

# PART B_______________________________________________________________________

test_metrics = evaluate_model(best_model, test_data, epsilon=0.4)

if test_metrics['non_rejected'] > 0:
    test_accuracy = (test_metrics['TP'] + test_metrics['TN']) / test_metrics['non_rejected']
    misclassification_loss = 1 - test_accuracy
else:
    test_accuracy = 0
    misclassification_loss = None

print("\nTest Data Results:")
print("Number of rejected samples:", test_metrics['no_class_count'])
print("Misclassification Loss (on non-rejected samples):", misclassification_loss)

test_metrics = evaluate_model(best_model, test_data, epsilon=0.499999999999999999)

if test_metrics['non_rejected'] > 0:
    test_accuracy = (test_metrics['TP'] + test_metrics['TN']) / test_metrics['non_rejected']
    misclassification_loss = 1 - test_accuracy
else:
    test_accuracy = 0
    misclassification_loss = None

print("\nTest Data Results:")
print("Number of rejected samples:", test_metrics['no_class_count'])
print("Misclassification Loss (on non-rejected samples):", misclassification_loss)
#_____________________________________________________________________________
#_____________________________________________________________________________

###############################################################################
###############################################################################
#
#  ██████╗       ████████╗██████╗ ███████╗███████╗
#  ██╔══██╗      ╚══██╔══╝██╔══██╗██╔════╝██╔════╝
#  ██║  ██║█████╗   ██║   ██████╔╝█████╗  █████╗  
#  ██║  ██║╚════╝   ██║   ██╔══██╗██╔══╝  ██╔══╝  
#  ██████╔╝         ██║   ██║  ██║███████╗███████╗
#  ╚═════╝          ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
#                                                 
###############################################################################
###############################################################################
res = oracle.q3_hyper(23607)
data = pd.read_csv('processed.cleveland.data')
data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','goal']

# Clean the data
data = data.replace('?', np.nan)

# numeric columns
fields = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
imputer = SimpleImputer(strategy='mean')
data[fields] = imputer.fit_transform(data[fields])

# Catetorical columns
fields = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
imputer = SimpleImputer(strategy='most_frequent')
data[fields] = imputer.fit_transform(data[fields])

# we have to check disease or no-disease, so make the goal column binary
data['goal'] = data['goal'].replace([1, 2, 3, 4], 1)    

# check if all are filled
print(data.isnull().sum())

# split the data
X = data.drop('goal', axis=1)
y = data['goal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

clf = DecisionTreeClassifier(random_state=69, criterion=res[0], splitter=res[1], max_depth=res[2])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# precision, accuracy, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
print(tabulate([[f'Accuracy: {accuracy:.4f}']], tablefmt="rounded_grid"))
report = classification_report(y_test, y_pred)
report_table = [row.split() for row in report.split('\n') if row]
print(report)

# Convert 'ca' and 'thal' to numeric
X_train['ca'] = pd.to_numeric(X_train['ca'], errors='coerce')
X_train['thal'] = pd.to_numeric(X_train['thal'], errors='coerce')

viz = dtreeviz.model(
    clf,
    X_train,
    y_train,
    target_name="Heart Disease",
    feature_names=X_train.columns.tolist(),
    class_names=["No Disease", "Disease"],
)

v = viz.view()
v.show()  

# Save the visualization to a specific location
v.save("decision_tree.svg")