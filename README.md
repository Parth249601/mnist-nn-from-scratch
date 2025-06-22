# 🧠 MNIST Digit Classifier — Neural Network from Scratch

This project implements a fully-connected neural network from scratch using only NumPy, to classify handwritten digits from the MNIST dataset.

### 🔧 Architecture
- Input layer: 784 neurons (28×28 flattened pixels)
- Hidden layer 1: 128 neurons + ReLU
- Hidden layer 2: 64 neurons + ReLU
- Output layer: 10 neurons + Softmax

### 🚀 Features
- Forward & backward propagation
- Mini-batch gradient descent
- He initialization
- Cross-entropy loss
- Accuracy reporting on training & test data

### 📊 Results
- Achieved **~97% train accuracy** and **~96% test accuracy** after 10 epochs.

### 🗂️ Directory Structure
mnist-nn-from-scratch/
│
├── data/                     # Raw MNIST files here
│
├── src/                     
│   └── model.py              # Neural Network Code
│
├── Papers/
    └── Neural_Network
    └── BackProp_EQNS
│
├── README.md                 # Project overview
