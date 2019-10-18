import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils

# cuda for GPU, cpu for CPU
mode = torch.device('cuda')

# Input file, first column are labels, rest are values for dimensions
data_file = "../data/1000exp20d.csv"

# Input file, first column are labels, rest are values for dimensions
data_file = "../data/1000exp20d.csv"

# Parameters
learn_rate = 1e-4
iterations = 5000
loss_margin = 50            # Margin for Contrastive Loss
pair_selection = "random"      # random, knn, or lsh

# Number of intermediate checkpoints to graph
checkpoints = 6

# Reading Data
print("Preparing Data")
data = []
labels = []
for ind, line in enumerate(open(data_file)):
    labels.append(np.array(int(line[0])))
    data.append(np.array(np.fromstring(line.split(',', 1)[1], sep = ',')))
data = np.column_stack(data)
labels = np.column_stack(labels)

# Numpy matrices to torch tensor of Float type
labels = torch.tensor(labels.transpose()).to(torch.float)
data = torch.tensor(data.transpose()).to(torch.float)

# Move to GPU if mode is CUDA
if mode == torch.device('cuda'):
    data = data.cuda()
    labels = labels.cuda()
    
# NN Architecture
print("Initializing Neural Net")
D_in, D_1, D_2, D_3, D_4, D_out = data.shape[1], 100, 50, 25, 10, 3

# Initial weights
w1 = torch.nn.init.xavier_uniform_(torch.randn(D_in, D_1, device=mode, requires_grad=True))
w2 = torch.nn.init.xavier_uniform_(torch.randn(D_1, D_2, device=mode, requires_grad=True))
w3 = torch.nn.init.xavier_uniform_(torch.randn(D_2, D_3, device=mode, requires_grad=True))
w4 = torch.nn.init.xavier_uniform_(torch.randn(D_3, D_4, device=mode, requires_grad=True))
w5 = torch.nn.init.xavier_uniform_(torch.randn(D_4, D_out, device=mode, requires_grad=True))

# Train Network
print("Commence Learning!!!")
accumulator = 0
losses = []
lsh_hash_ptr = 0
checkpoint_count = 0
for it in range(iterations):
    # Forward Pass
    results = data.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3).clamp(min=0).mm(w4).clamp(min=0).mm(w5)
    
    #Loss Calculation
    if pair_selection == 'random':    
        if_same, euclidean_dist = utils.rand_pairs(results, labels)
    elif pair_selection == 'knn':
        if_same, euclidean_dist = utils.knn_pairs(results, labels, it, mode)
    elif pair_selection == 'lsh':
        if_same, euclidean_dist, lsh_hash = utils.lsh_pairs(results, labels, it, lsh_hash_ptr)
        
    loss = torch.mean((if_same) * torch.pow(euclidean_dist, 2) +
            (1 - if_same) * torch.pow(torch.clamp(loss_margin - euclidean_dist, min=0), 2))
    
    accumulator += loss.item()

    #Backward Pass
    loss.backward()
    
    with torch.no_grad():
        # Update Gradients
        w1 -= learn_rate * w1.grad
        w2 -= learn_rate * w2.grad
        w3 -= learn_rate * w3.grad
        w4 -= learn_rate * w4.grad
        w5 -= learn_rate * w5.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_(); w2.grad.zero_(); w3.grad.zero_(); w4.grad.zero_(); w5.grad.zero_()

    if it % 100 == 0:
        if it == 0:
            print("Iteration:", it, "\t|\tInitial Loss:", round(accumulator))
            losses.append(accumulator)
        else:
            avg_loss = accumulator / 100
            print("Iteration:", it, "\t|\tAverage Loss (last 100):", round(avg_loss))
            losses.append(avg_loss)
        accumulator = 0

    if it % round(iterations / (checkpoints + 1)) == 0:
        utils.graph_points(results, labels, pair_selection + "_" + str(checkpoint_count))
        checkpoint_count += 1

utils.graph_points(results, labels, pair_selection + "_final")
utils.graph_loss(losses, pair_selection)