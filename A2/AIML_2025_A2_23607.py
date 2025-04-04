import oracle
import time
import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pickle
import pandas as pd
from cvxopt import matrix, solvers


#Saving the data 
# res = oracle.q1_get_cifar100_train_test(23607)
# train = res[0]
# test= res[1]
# train_data = []
# train_labels = []
# for i in range(len(train)):
#     train_data.append(train[i][0])
#     train_labels.append(train[i][1])

# test_data = []
# test_labels = []
# for i in range(len(test)):
#     test_data.append(test[i][0])
#     test_labels.append(test[i][1])

# train_data = np.array(train_data)
# train_labels = np.array(train_labels)
# test_data = np.array(test_data)
# test_labels = np.array(test_labels)

# np.save('train_data.npy', train_data)
# np.save('train_labels.npy', train_labels)
# np.save('test_data.npy', test_data)
# np.save('test_labels.npy', test_labels)

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

##########################################
# PERCEPTRON
##########################################
R = np.max(np.linalg.norm(train_data, axis=1))
def perceptron(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])
    norm_history = []
    mistakes = 0
    
    for _ in range(max_iter):
        converged = True
        for i in range(len(X)):
            if y[i] * np.dot(w, X[i]) <= 0:
                w += y[i] * X[i]
                mistakes += 1
                converged = False
        norm_history.append(np.linalg.norm(w))
        if converged:
            print('Converged after', _, 'iterations')
            break
    return w, mistakes, norm_history
    # was using the return values to debug, so didn't modify further. Can omit mistakes and norm_history tho.

X = train_data
y = train_labels

# was using it to visualize if the data is linearly separable (1D)
def FLD_plot(X, y):
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X, y)

    plt.scatter(X_lda[y == 1], np.zeros_like(X_lda[y == 1]), color='blue', label='Class +1')
    plt.scatter(X_lda[y == -1], np.zeros_like(X_lda[y == -1]), color='red', label='Class -1')
    plt.axvline(0, color='green', linestyle='--', label='Fisher Decision Boundary')
    plt.xlabel('LDA Dimension 1')
    plt.ylabel('Projection')
    plt.legend()
    plt.title('Fisher LDA with Decision Boundary in 1D')
    plt.show()

FLD_plot(X, y) # Showed heavy overlap. Quite obvious.

def mistakes_perceptron(w, X, y):
    mistakes = 0
    for i in range(len(X)):
        if y[i] * np.dot(w, X[i]) <= 0:
            mistakes += 1
    return mistakes

# number of mistakes plot
max_iters = [100, 200, 500, 1000, 2000, 5000, 7500, 10000]
mistakes_train = []
mistakes_test = []
for max_iter in max_iters:
    w, _, _ = perceptron(train_data, train_labels, max_iter)
    mistakes_train.append(mistakes_perceptron(w, train_data, train_labels))
    mistakes_test.append(mistakes_perceptron(w, test_data, test_labels))

# Misclassification rate plot
# the following data was obtained after running the perceptron function for different values of max_iter
# I just copied the jupyter notebook output and plotted separately later.
mistakes_train__ = [354, 268, 296, 350, 388, 337, 405, 285, 293, 375, 353, 376, 356]
mistakes_test__=[65, 42, 53, 69, 80, 66, 81, 55, 60, 80, 69, 79, 70]
max_iters__ = [100, 200, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000]
len_train = len(train_data)
len_test = len(test_data)
mistakes_train__ = [m/len_train for m in mistakes_train__]
mistakes_test__ = [m/len_test for m in mistakes_test__]

plt.plot(max_iters__, mistakes_train__, label='Train')
plt.plot(max_iters__, mistakes_test__, label='Test')
plt.xlabel('Max Iterations')
plt.ylabel('Misclassification Rate')
plt.title('Perceptron Misclassification Rate vs Iterations')
plt.legend()
plt.show()

##########################################
# LINEAR SVM
##########################################
def mistakes_svm(w, b, X, y):
    mistakes = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            mistakes += 1
    return mistakes

X = train_data
y = train_labels

# Primal SVM
##########################################
C = 20
n_samples, n_features = X.shape

# Primal SVM matrices
P = matrix(np.block([
    [np.eye(n_features), np.zeros((n_features, n_samples + 1))],
    [np.zeros((n_samples + 1, n_features + n_samples + 1))]
]))
q = matrix(np.hstack([np.zeros(n_features), np.zeros(1), C * np.ones(n_samples)]))
G_top = np.hstack([-y[:, np.newaxis] * X, -y[:, np.newaxis], -np.eye(n_samples)])
G_bottom = np.hstack([np.zeros((n_samples, n_features)), np.zeros((n_samples, 1)), -np.eye(n_samples)])
G = matrix(np.vstack([G_top, G_bottom]))
h = matrix(np.hstack([-np.ones(n_samples), np.zeros(n_samples)]))

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)
w = np.array(sol['x'][:n_features]).flatten()
b = sol['x'][n_features]

# print('Primal SVM')
# print(mistakes_svm(w, b, train_data, train_labels)) => 227
# print(mistakes_svm(w, b, test_data, test_labels)) => 40

#DUAL SVM
##########################################
C = 20
n_samples, n_features = X.shape
K = np.dot(X, X.T) * np.outer(y, y)
P_dual = matrix(K)
q_dual = matrix(-np.ones(n_samples))
G_dual = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
h_dual = matrix(np.hstack([np.zeros(n_samples), C * np.ones(n_samples)]))
A_dual = matrix(y.reshape(1, -1), (1, n_samples), 'd')
b_dual = matrix(0.0)

solvers.options['show_progress'] = False
sol = solvers.qp(P_dual, q_dual, G_dual, h_dual, A_dual, b_dual)
alpha = np.ravel(sol['x'])
sv = alpha > 1e-5
ind = np.arange(len(alpha))[sv]
alpha_sv = alpha[sv]
X_sv = X[sv]
y_sv = y[sv]

w = np.sum(alpha_sv[:, np.newaxis] * y_sv[:, np.newaxis] * X_sv, axis=0)
b = np.mean(y_sv - np.dot(X_sv, w))

# print('Dual SVM')
# print(mistakes_svm(w, b, train_data, train_labels)) => 232
# print(mistakes_svm(w, b, test_data, test_labels)) => 34
##########################################
#mistakes file saving
mistakes_idx = []
for i in range(len(train_data)):
    if train_labels[i] * np.dot(w, train_data[i]) <= 0:
        mistakes_idx.append(i)
with open('inseparable_23607.csv', 'w') as f:
    for idx in mistakes_idx:
        f.write(str(idx) + ',')

##########################################
# Plotting times for different values of C
np.random.seed(42)
n_samples, n_features = 1000, 27
X = np.random.randn(n_samples, n_features)
y = np.random.choice([-1, 1], size=n_samples)

C_values = [0.1, 1, 10, 20, 30, 50, 70, 100, 200, 300, 400, 500, 1000]

primal_times = []
dual_times = []

for C in C_values:
    start = time.time()

    P = matrix(np.block([
        [np.eye(n_features), np.zeros((n_features, n_samples + 1))],
        [np.zeros((n_samples + 1, n_features + n_samples + 1))]
    ]))
    
    q = matrix(np.hstack([np.zeros(n_features), np.zeros(1), C * np.ones(n_samples)]))

    G_top = np.hstack([-y[:, np.newaxis] * X, -y[:, np.newaxis], -np.eye(n_samples)])
    G_bottom = np.hstack([np.zeros((n_samples, n_features)), np.zeros((n_samples, 1)), -np.eye(n_samples)])
    G = matrix(np.vstack([G_top, G_bottom]))
    h = matrix(np.hstack([-np.ones(n_samples), np.zeros(n_samples)]))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    primal_time = time.time() - start
    primal_times.append(primal_time)

    start = time.time()
    K = np.dot(X, X.T) * np.outer(y, y)
    
    P_dual = matrix(K)
    q_dual = matrix(-np.ones(n_samples))
    
    G_dual = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
    h_dual = matrix(np.hstack([np.zeros(n_samples), C * np.ones(n_samples)]))
    
    A_dual = matrix(y.reshape(1, -1), (1, n_samples), 'd')
    b_dual = matrix(0.0)

    sol = solvers.qp(P_dual, q_dual, G_dual, h_dual, A_dual, b_dual)

    dual_time = time.time() - start
    dual_times.append(dual_time)

    print(f"C={C}, Primal Time: {primal_time:.4f}s, Dual Time: {dual_time:.4f}s")

plt.figure(figsize=(12, 6))
plt.plot(C_values, primal_times, label="Primal SVM", marker='o', color='blue')
plt.plot(C_values, dual_times, label="Dual SVM", marker='x', color='red')

plt.xlabel("C Value (Regularization Parameter)")
plt.ylabel("Time (seconds)")
plt.title("Primal vs Dual SVM Solution Time")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.show()
#OUTPUT
# C=0.1, Primal Time: 1.8242s, Dual Time: 1.7066s
# C=1, Primal Time: 2.0232s, Dual Time: 1.8078s
# C=10, Primal Time: 1.9438s, Dual Time: 1.8205s
# C=20, Primal Time: 1.9472s, Dual Time: 1.8466s
# C=30, Primal Time: 1.9961s, Dual Time: 1.9180s
# C=50, Primal Time: 1.8907s, Dual Time: 1.7220s
# C=70, Primal Time: 1.7664s, Dual Time: 1.6631s
# C=100, Primal Time: 1.8814s, Dual Time: 1.8018s
# C=200, Primal Time: 1.8766s, Dual Time: 1.7341s
# C=300, Primal Time: 1.8367s, Dual Time: 1.8365s
# C=400, Primal Time: 1.8924s, Dual Time: 1.8450s
# C=500, Primal Time: 1.9386s, Dual Time: 1.8165s
# C=1000, Primal Time: 1.9492s, Dual Time: 1.7995s
##########################################

##########################################
# GAUSSIAN SVM
##########################################
X = train_data
y = train_labels
C = 60  
gamma = 0.1  
num_support_vectors = []

def rbf_kernel(X1, X2, gamma):
    sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

K = rbf_kernel(X, X, gamma)
K = np.outer(y, y) * K 

P = matrix(K)
q = matrix(-np.ones(X.shape[0]))
G = matrix(np.vstack((-np.eye(X.shape[0]), np.eye(X.shape[0]))))
h = matrix(np.hstack((np.zeros(X.shape[0]), np.ones(X.shape[0]) * C)))
A = matrix(y.reshape(1, -1).astype(float))
b = matrix(np.zeros(1))

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])
sv = (alphas > 1e-5) & (alphas < C)
print(f"Number of support vectors: {np.sum(sv)} |", "C:", C)
num_support_vectors.append(np.sum(sv))
w = np.sum(alphas[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
support_vector_idx = np.where(sv)[0][0]
b = y[support_vector_idx] - np.sum(alphas * y * rbf_kernel(X, X[support_vector_idx:support_vector_idx+1], gamma))

##########################################
# Counting misclassifications
misclassified_indices = []
with open ('inseparable_23607.csv', 'r') as f:
    for line in f:
        misclassified_indices = line.split(',')
    
# remove misclassified indices from the train data
misclassified_indices = np.array([int(idx) for idx in misclassified_indices if idx != ''])
ls_X = np.delete(train_data, misclassified_indices, axis=0)
ls_y = np.delete(train_labels, misclassified_indices)

w,_, _ = perceptron(ls_X, ls_y, max_iter=1000)
mistakes = mistakes_perceptron(w, test_data, test_labels)
# => Converged after 875 iterations
#Plotting
misclassified_indices = []
with open ('inseparable_23607.csv', 'r') as f:
    for line in f:
        misclassified_indices = line.split(',')
    
# remove misclassified indices from the train data
misclassified_indices = np.array([int(idx) for idx in misclassified_indices if idx != ''])
ls_X = np.delete(train_data, misclassified_indices, axis=0)
ls_y = np.delete(train_labels, misclassified_indices)

iters = [i*50 for i in range(19)]
mistakes_train = []
for max_iter in iters:
    w, _, _ = perceptron(ls_X, ls_y, max_iter)
    mistakes_train.append(mistakes_perceptron(w, ls_X, ls_y))
plt.plot(iters, mistakes_train)
plt.xlabel('Iterations')
plt.ylabel('Mistakes')
plt.title('Perceptron Mistakes vs Iterations (misclassified points removed) \n Cvg after 875 iterations')
plt.show()
##########################################
##########################################
# QUESTION 2
##########################################
##########################################
# Dataloader class
data_dir = "q2_data"

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=(28, 28), flatten=True):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.flatten = flatten

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32) / 255.0  
        image = (image - 0.5) / 0.5                        
        if self.flatten:
            image = image.flatten() 
            image_tensor = torch.tensor(image, dtype=torch.float32)
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor

def load_dataset(data_dir, image_size=(28, 28), flatten=True, train_ratio=0.8):
    image_paths = []
    labels = []

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, img_file))
                labels.append(int(label))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    n_samples = len(image_paths)
    n_train = int(n_samples * train_ratio)

    indices = np.random.permutation(n_samples)
    train_indices, test_indices = indices[:n_train], indices[n_train:]
    train_dataset = ImageDataset(image_paths[train_indices], labels[train_indices], image_size, flatten)
    test_dataset = ImageDataset(image_paths[test_indices], labels[test_indices], image_size, flatten)

    return train_dataset, test_dataset

##########################################
# MLP
##########################################
dataset_mlp, dataset_mlp_test = load_dataset(data_dir, flatten=True)
dataloader_mlp = DataLoader(dataset_mlp, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader_mlp:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved successfully!")

model = MLP().to(device)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

y_true = []
y_pred = []

for images, labels in dataloader_mlp:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

"""
OUTPUT:
Using device: cuda
Epoch [1/5], Loss: 86.7187, Accuracy: 77.81%
Epoch [2/5], Loss: 44.0557, Accuracy: 89.33%
Epoch [3/5], Loss: 32.5295, Accuracy: 91.69%
Epoch [4/5], Loss: 27.2577, Accuracy: 93.26%
Epoch [5/5], Loss: 19.5647, Accuracy: 95.28%
Model saved successfully!
Confusion Matrix:
[[796   0   0   0   0   3   0   0   5   2]
 [  0 771   2   2   1   1   0   1   4   6]
 [  0   0 757   6   5   0   1   7   7   2]
 [  0   4   4 752   0   9   0   1  14   7]
 [  0   4   2   0 740   0   0   2   0  44]
 [  1   2   0   5   0 791   0   0  16   4]
 [ 11   3   7   1  18  20 716   0  21   1]
 [  1   3   4   1   1   3   0 788   3  20]
 [  0   1   2   2   1   1   0   0 784   2]
 [  1   0   0   4   2   0   0   4   5 788]]
"""
# Test
test_set = dataset_mlp_test
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
model.eval()
y_true = []
y_pred = []

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred))
"""
OUTPUT:
Confusion Matrix:
[[188   0   1   1   0   3   0   0   1   0]
 [  0 211   0   0   0   0   0   0   0   1]
 [  2   2 193   5   4   0   1   2   6   0]
 [  0   0   1 193   0   4   0   5   3   3]
 [  0   0   1   0 195   0   1   1   0  10]
 [  1   1   0   7   0 164   2   0   3   3]
 [  3   3   3   0   3   8 178   0   4   0]
 [  0   0   1   1   0   0   0 161   0  13]
 [  1   2   0   1   1   2   0   0 197   3]
 [  0   0   0   4   3   1   0   1   2 185]]
Accuracy: 0.9325
...
"""
# Function for testing a single image
def test_image_mlp(path, model, flatten = True):
    image = Image.open(path).convert("L")
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.float32) / 255.0  
    image = (image - 0.5) / 0.5
    image = image.flatten()
    image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Predicted label: {predicted.item()}")
    probs = torch.softmax(output, dim=1)
    probs = [f"{p:.4f}" for p in probs.squeeze().tolist()]
    return predicted, probs
"""
p, prob = test_image_mlp("q2_data/3/50.jpg", model)
print(f"Class probabilities: {prob}")
>>Predicted label: 3
  Class probabilities: ['0.0008', '0.0006', '0.0007', '0.9840', '0.0000', '0.0124', '0.0000', '0.0000', '0.0015', '0.0001']
"""
##########################################
# CNN
##########################################
dataset_cnn, dataset_cnn_test = load_dataset(data_dir, flatten=False)
dataloader_cnn = DataLoader(dataset_cnn, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  
            nn.ReLU(),
            nn.Linear(128, 10)   
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader_cnn:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "cnn_model.pth")
print("CNN model saved successfully!")

model = CNN().to(device)
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

y_true = []
y_pred = []

for images, labels in dataloader_cnn:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
"""
OUTPUT:
Using device: cuda
Epoch [1/5], Loss: 77.9326, Accuracy: 80.89%
Epoch [2/5], Loss: 16.7518, Accuracy: 96.10%
Epoch [3/5], Loss: 10.6209, Accuracy: 97.36%
Epoch [4/5], Loss: 8.0972, Accuracy: 98.14%
Epoch [5/5], Loss: 6.5579, Accuracy: 98.55%
CNN model saved successfully!
Confusion Matrix:
[[786   0   1   0   0   1   1   2   0   0]
 [  1 822   1   0   0   0   0   0   0   1]
 [  0   2 790   1   1   0   1   4   3   0]
 [  0   0   0 785   0   1   0   1   0   0]
 [  0   1   0   0 806   0   0   2   0   1]
 [  0   0   0   2   0 785   0   0   0   0]
 [  1   0   0   0   0  10 784   0   1   0]
 [  0   2   0   0   1   0   0 789   0   0]
 [  0   0   0   1   0   4   0   0 792   0]
 [  0   0   0   5   1   3   0   7   2 795]]
"""
# Test
test_set = dataset_cnn_test
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# CNN model is model
model.eval()
y_true = []
y_pred = []

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
"""
OUTPUT:
Confusion Matrix:
[[204   0   0   0   0   1   3   1   0   0]
 [  0 170   0   1   1   0   0   3   0   0]
 [  0   0 193   0   0   0   0   2   3   0]
 [  1   0   0 203   0   3   0   1   2   3]
 [  0   2   0   0 182   0   2   1   2   1]
 [  0   0   0   1   0 212   0   0   0   0]
 [  0   0   0   0   0   2 201   0   1   0]
 [  0   0   2   0   0   0   0 203   1   2]
 [  1   0   1   2   0   1   1   1 195   1]
 [  0   0   0   2   4   1   0   4   0 176]]
"""
# Function for testing a single image
def test_image_cnn(path, model):
    image = Image.open(path).convert("L")   
    image = image.resize((28, 28))   
    image = np.array(image, dtype=np.float32) / 255.0  
    image = (image - 0.5) / 0.5               

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

        probs = torch.softmax(output, dim=1).squeeze().tolist()
        probs = [f"{p:.4f}" for p in probs]

    print(f"Predicted label: {predicted.item()}")
    return predicted.item(), probs
"""
p, prob = test_image_cnn("q2_data/3/50.jpg", model)
print(f"Class probabilities: {prob}")
>>Predicted label: 3
  Class probabilities: ['0.0000', '0.0000', '0.0000', '1.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000']
"""
##########################################
# PCA
##########################################
image_size = (28, 28)
data = []
labels = []

def preprocess_img(img):
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.flatten() 
    return img_array

def postprocess_img(img_array):
    img_array = img_array * 255.0
    img_array = img_array.reshape(image_size)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

for label in sorted(os.listdir(data_dir)):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image = Image.open(img_path)
            img_array = preprocess_img(image)
            data.append(img_array)
            labels.append(int(label))

data = np.array(data)
labels = np.array(labels)

mean = np.mean(data, axis=0)
data_centered = data - mean

C = np.cov(data_centered.T)
eigenvalues, eigenvectors = np.linalg.eig(C)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Image reconstruction
ks = [1,2,5,8, 10,15,20, 50, 100, 400, 784]
img_path = "q2_data/3/50.jpg"
image = Image.open(img_path)
image = np.array(image, dtype=np.float32) / 255.0
image = image.flatten()
image_centered = image - mean

for k in ks:
    P = eigenvectors[:, :k]
    projected_image = np.dot(image_centered, P)

    reconstructed_image = np.dot(projected_image, P.T) + mean 
    reconstructed_image = postprocess_img(reconstructed_image)
    reconstructed_image.save(f"PCA_reconstructed_img/{k}.jpg")
image = postprocess_img(image)
image.save("PCA_reconstructed_img/original.jpg")
##########################################
# MLP with PCA
##########################################
k = 100
P = eigenvectors[:, :k]
projected_data = np.dot(data, P)

class PCAImageDataset(Dataset):
    def __init__(self, projected_data, labels):
        self.projected_data = projected_data
        self.labels = labels

    def __len__(self):
        return len(self.projected_data)

    def __getitem__(self, idx):
        image_tensor = torch.tensor(self.projected_data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor

data_dir = "q2_data"
image_size = (28, 28)

dataset_pca = PCAImageDataset(projected_data, labels)
dataloader_pca = DataLoader(dataset_pca, batch_size=64, shuffle=True)

class MLP_PCA(nn.Module):
    def __init__(self, input_dim=k):
        super(MLP_PCA, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) 
        )

    def forward(self, x):
        return self.model(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_pca = MLP_PCA(input_dim=k).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pca.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader_pca:
        images, labels = images.to(device), labels.to(device)

        outputs = model_pca(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model_pca.state_dict(), "mlp_pca_model.pth")
print("MLP with PCA model saved successfully!")

model_pca = MLP_PCA(input_dim=k).to(device)
model_pca.load_state_dict(torch.load("mlp_pca_model.pth"))
model_pca.eval()

y_true = []
y_pred = []

for images, labels in dataloader_pca:
    images, labels = images.to(device), labels.to(device)

    outputs = model_pca(images)
    _, predicted = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
##########################################
# Logostic Regression with PCA
##########################################
image_size = (28, 28)
data = []
labels = []

def preprocess_img(img):
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.flatten()  
    return img_array

data_dir = "q2_data"
for label in sorted(os.listdir(data_dir)):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(data_dir, label, img_file)
            image = Image.open(img_path)
            img_array = preprocess_img(image)
            data.append(img_array)
            labels.append(int(label))

data = np.array(data)
labels = np.array(labels)

scaler = StandardScaler()
data = scaler.fit_transform(data)

m = np.mean(data, axis=0)
data -= m
C = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(C)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

k = 100
P = eigenvectors[:, :k]
pca_data = np.dot(data, P)
X_train, X_test, y_train, y_test = train_test_split(pca_data, labels, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, tol=1e-5):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n + 1)
        X = np.hstack((X, np.ones((m, 1)))) 

        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            predictions = sigmoid(z)

            gradient = np.dot(X.T, (predictions - y)) / m
            self.theta -= self.lr * gradient

            if np.linalg.norm(gradient) < self.tol:
                break

    def predict_proba(self, X):
        m = X.shape[0]
        X = np.hstack((X, np.ones((m, 1))))  
        return sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

print("\nTraining One-vs-Rest Binary Classifiers...")

num_classes = len(np.unique(y_train))
binary_classifiers = {}

for c in range(num_classes):
    print(f"Training binary classifier for class {c}...")
    
    y_binary = (y_train == c).astype(int)

    clf = LogisticRegression(lr=0.01, epochs=2000)
    clf.fit(X_train, y_binary)

    binary_classifiers[c] = clf

print("\nGenerating ROC Curves (One-vs-Rest)...")
plt.figure(figsize=(12, 8))

for c in range(num_classes):
    print(f"Generating ROC for class {c}...")
    y_test_binary = (y_test == c).astype(int)
    y_prob = binary_classifiers[c].predict_proba(X_test)

    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {c} (AUC = {roc_auc:.2f})')

# Plot one-vs-rest
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for One-vs-Rest Binary Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("\nMulticlass ROC and AUC:")

y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
y_train_bin = label_binarize(y_train, classes=np.arange(num_classes))
y_prob_multiclass = np.zeros((X_test.shape[0], num_classes))

for c in range(num_classes):
    y_prob_multiclass[:, c] = binary_classifiers[c].predict_proba(X_test)

# ROC and AUC for multiclass
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(12, 8))

# ROC curve for each class
for c in range(num_classes):
    fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, c], y_prob_multiclass[:, c])
    roc_auc[c] = auc(fpr[c], tpr[c])

    plt.plot(fpr[c], tpr[c], label=f'Class {c} (AUC = {roc_auc[c]:.2f})')
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob_multiclass.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

# Plot multiclass
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves')
plt.legend(loc='lower right')
plt.grid()
plt.show()

y_true = y_test
y_pred = np.argmax(y_prob_multiclass, axis=1)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
##########################################
##########################################
# QUESTION 3
##########################################
##########################################
# I saved the data in a pickle file
data1 = pickle.load(open('data1.pkl', 'rb'))
data2 = pickle.load(open('data2.pkl', 'rb'))
X1_train = np.array(data1[0])
y1_train = np.array(data1[1])
X1_test = np.array(data1[2])
y1_test = np.array(data1[3])

X2_train = np.array(data2[0])
y2_train = np.array(data2[1])
X2_test = np.array(data2[2])
y2_test = np.array(data2[3])
# each column of X is a sample and last row is 1
X1_train = X1_train.T
X1_test = X1_test.T
X2_train = X2_train.T
X2_test = X2_test.T
X1_train = np.vstack((X1_train, np.ones(X1_train.shape[1])))
X1_test = np.vstack((X1_test, np.ones(X1_test.shape[1])))
X2_train = np.vstack((X2_train, np.ones(X2_train.shape[1])))
X2_test = np.vstack((X2_test, np.ones(X2_test.shape[1])))

def linear_regression(X, y):
    xxt = np.dot(X, X.T)
    # if inverse exists
    if np.linalg.matrix_rank(xxt) == xxt.shape[0]:
        xxt_inv = np.linalg.inv(xxt)
        print('inverse exists')
    else:
        xxt_inv = np.linalg.pinv(xxt)
        print('inverse does not exist')

    w = np.dot(xxt_inv, np.dot(X, y))
    return w

def ridge_regression(X, y, alpha):
    xxt = np.dot(X, X.T)
    xy = np.dot(X, y)
    I = np.eye(xxt.shape[0])
    w = np.dot(np.linalg.inv(xxt + alpha * I), xy)
    return w

# Lin reg calculation
w1 = linear_regression(X1_train, y1_train)
w2 = linear_regression(X2_train, y2_train)
mse1 = np.mean((np.dot(w1.T, X1_test) - y1_test) ** 2)
mse2 = np.mean((np.dot(w2.T, X2_test) - y2_test) ** 2)

print("\nOLS")
print('mse1 (D1):', mse1)
print('mse2 (D2):', mse2)

print('w1 OLS:', w1.T)
# print('w2 OLS:', w2.T)
np.savetxt("w_ols_23607.csv", w2.T, delimiter = ",")
"""
inverse exists
inverse does not exist

OLS
mse1 (D1): 2.7685343512966045
mse2 (D2): 53.852755961317115
w1 OLS: [[0.35655766 0.32183454 0.00351676 0.8842572  0.16073379 0.07412291]]
"""

# Ridge reg calculation
alpha = 100
w1 = ridge_regression(X1_train, y1_train, alpha)
w2 = ridge_regression(X2_train, y2_train, alpha)
# calculate the mean squared error on the test set
mse1 = np.mean((np.dot(w1.T, X1_test) - y1_test) ** 2)
mse2 = np.mean((np.dot(w2.T, X2_test) - y2_test) ** 2)

print("RIDGE REGRESSION")
print('mse1 (D1):', mse1)
print('mse2 (D2):', mse2)

print('w1 Ridge:', w1.T)
# print('w2 Ridge:', w2.T)
np.savetxt("w_rr_23607.csv", w2.T, delimiter = ",")
"""
RIDGE REGRESSION
mse1 (D1): 1.6587716233420446
mse2 (D2): 43.93170321794892
w1 Ridge: [[ 0.10344289  0.04432345 -0.02761292  0.13022108  0.0014239   0.00560629]]
"""
# Plot alpha vs MSE
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
mse1_list = []
mse2_list = []

for alpha in alphas:
    w1 = ridge_regression(X1_train, y1_train, alpha)
    w2 = ridge_regression(X2_train, y2_train, alpha)
    # calculate the mean squared error on the test set
    mse1 = np.mean((np.dot(w1.T, X1_test) - y1_test) ** 2)
    mse2 = np.mean((np.dot(w2.T, X2_test) - y2_test) ** 2)

    # print('mse1:', mse1)
    # print('mse2:', mse2)
    mse1_list.append(mse1)
    mse2_list.append(mse2)

import matplotlib.pyplot as plt
# plt.plot(alphas, mse1_list, label='data1')
plt.plot(alphas, mse2_list, label='data2')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('mean squared error')
plt.legend()
plt.show()
##########################################
#Q 3.2
##########################################
print(oracle.q3_stocknet(23607)) # AEP
df = pd.read_csv('AEP.csv')
df = df['Close']
df_original = df.copy()
scaler = StandardScaler()
df = scaler.fit_transform(df.values.reshape(-1, 1))
df = df.flatten()

def get_data(df, t):
    n = len(df)
    X = np.zeros((n-t, t))
    for i in range(n-t):
        X[i] = df[i:i+t]
    Y = df[t:]
    l = len(X)
    split = int(0.8*l)
    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    return X_train, X_test, Y_train, Y_test

def plot_linear_svr(X_train, Y_train, X_test, Y_test, w, b, t, df_original, scaler):

    Y_pred_scaled = X_test @ w + b
    Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    Y_pred_scaled = -Y_pred_scaled  
    Y_min, Y_max = df_original.min(), df_original.max()
    Y_pred = Y_min + (Y_pred_scaled - Y_pred_scaled.min()) * (Y_max - Y_min) / (Y_pred_scaled.max() - Y_pred_scaled.min())

    prev_avg = []
    train_end = len(Y_train)

    for i in range(len(Y_test)):
        avg_window_start = max(0, train_end - t + i)
        avg_window_end = train_end + i
        prev_avg.append(np.mean(df_original[avg_window_start:avg_window_end]))

    prev_avg = np.array(prev_avg)

    plt.figure(figsize=(14, 6))
    plt.plot(Y_test_original, label='Actual Closing Price', color='blue', linewidth=0.7)
    plt.plot(Y_pred, label='Predicted Closing Price', color='green', linestyle='--', linewidth=1)
    plt.plot(prev_avg, label=f'Previous {t}-Day Average', color='orange', linestyle='-.', linewidth=1.5)
    
    plt.xlabel('Time (Days)')
    plt.ylabel('Price')
    plt.title(f'linear SVR Prediction vs Actual vs Previous {t}-Day Average')
    plt.ylim(0, max(np.max(Y_test_original), np.max(Y_pred), np.max(prev_avg)) * 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()



def linear_svr(X, Y):
    C = 1.0  
    epsilon = 0.1
    n, t = X.shape
    K = X @ X.T
    P = np.block([
        [K, -K],
        [-K, K]
    ])
    q = np.hstack([epsilon + Y, epsilon - Y])
    G = np.vstack([
        np.eye(2 * n),
        -np.eye(2 * n)
    ])
    h = np.hstack([C * np.ones(2 * n), np.zeros(2 * n)])
    A = np.hstack([np.ones(n), -np.ones(n)]).reshape(1, -1)
    b = np.array([0.0])
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    alpha = np.array(sol['x']).flatten()[:n]
    alpha_star = np.array(sol['x']).flatten()[n:]

    w = np.sum((alpha - alpha_star).reshape(-1, 1) * X, axis=0)
    b = np.mean(Y - X @ w)
    return w, b

def kernel_rbf(x1, x2, gamma):
    sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * sq_dist)

def kernel_svr_dual(X_train, Y_train, gamma, C=1.0, epsilon=0.1):
    n = len(Y_train)
    K = kernel_rbf(X_train, X_train, gamma)
    P = np.block([[K, -K], [-K, K]])  # Kernel matrix for both alpha and alpha*
    q = np.hstack([epsilon + Y_train, epsilon - Y_train])
    G = np.vstack([
        np.eye(2 * n), 
        -np.eye(2 * n)
    ])
    h = np.hstack([
        np.ones(2 * n) * C,  # Upper bound (C)
        np.zeros(2 * n)       # Lower bound (0)
    ])
    A = np.hstack([np.ones(n), -np.ones(n)]).reshape(1, -1)
    b = np.array([0.0])
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()
    alpha = alphas[:n]
    alpha_star = alphas[n:]

    w = np.dot((alpha - alpha_star), K)
    support_vector_indices = np.where((alpha > 1e-5) | (alpha_star > 1e-5))[0]
    if len(support_vector_indices) > 0:
        b = np.mean(
            Y_train[support_vector_indices] 
            - np.dot((alpha - alpha_star), K[:, support_vector_indices]))
    else:
        b = 0.0
    return alpha, alpha_star, w, b

def plot_rbf_svr(X_train, Y_train, X_test, Y_test, alpha, alpha_star, b, gamma, t, df_original, scaler):
    K_test = kernel_rbf(X_train, X_test, gamma)
    Y_pred_scaled = np.dot((alpha - alpha_star), K_test) + b
    Y_pred_scaled = -Y_pred_scaled
    Y_pred = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
    Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    prev_avg = []
    train_end = len(Y_train)

    for i in range(len(Y_test)):
        avg_window_start = max(0, train_end - t + i)
        avg_window_end = train_end + i
        prev_avg.append(np.mean(df_original[avg_window_start:avg_window_end]))

    prev_avg = np.array(prev_avg)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(Y_test_original, label='Actual Closing Price', color='blue', linewidth=0.8)
    plt.plot(Y_pred, label='Predicted Closing Price', color='green', linestyle='--', linewidth=0.8)
    plt.plot(prev_avg, label=f'Previous {t}-Day Average', color='orange', linestyle='-.', linewidth=0.8)
    
    plt.xlabel('Time (Days)')
    plt.ylabel('Price (Original Scale)')
    plt.title(f'RBF SVR Prediction vs Actual vs Previous {t}-Day Average | γ = {gamma}')
    
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

# Linear SVRs
ts = [7, 30, 90]
for t in ts:
    X_train, X_test, Y_train, Y_test = get_data(df, t)
    w, b = linear_svr(X_train, Y_train)
    print("T = ", t)
    plot_linear_svr(X_train, Y_train, X_test, Y_test, w, b, t, df_original, scaler)

# RBF SVRs
gammas = [1,0.1, 0.01, 0.001]
ts = [7, 30, 90]
for t in ts:
    for gamma in gammas:
        X_train, X_test, Y_train, Y_test = get_data(df, t)
        alpha, alpha_star, w, b = kernel_svr_dual(X_train, Y_train, gamma)
        print("T = ", t, "Gamma = ", gamma)
        plot_rbf_svr(X_train, Y_train, X_test, Y_test, alpha, alpha_star, b, gamma, t, df_original, scaler)

##########################################
##########################################



