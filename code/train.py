import torch
import numpy as np
import utils

mode = torch.device('cuda')

# Parameters
learn_rate = 1e-12
iterations = 5000
loss_margin = 500           # Margin for Contrastive Loss
graph_freq = 50             # How often to generate data for graphing
pair_selection = "random"   # random, knn, or lsh

# Reading Data
print("Preparing Data")
data = []
labels = []
for ind, line in enumerate(open("../data/data.csv")):
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
D_in, D_1, D_2, D_3, D_4, D_out = data.shape[1], 100, 50, 25, 10, 3

# Randomly Initialize Weights
w1 = torch.randn(D_in, D_1, device=mode, requires_grad=True)
w2 = torch.randn(D_1, D_2, device=mode, requires_grad=True)
w3 = torch.randn(D_2, D_3, device=mode, requires_grad=True)
w4 = torch.randn(D_3, D_4, device=mode, requires_grad=True)
w5 = torch.randn(D_4, D_out, device=mode, requires_grad=True)

print("Commence Learning!!!")
accumulator = 0
losses = []
for it in range(iterations):
    # Forward Pass
    h1 = data.mm(w1)
    h1_relu = h1.clamp(min=0)

    h2 = h1_relu.mm(w2)
    h2_relu = h2.clamp(min=0)

    h3 = h2_relu.mm(w3)
    h3_relu = h3.clamp(min=0)

    h4 = h3_relu.mm(w4)
    h4_relu = h4.clamp(min=0)

    results = h4_relu.mm(w5)
    
    #print(results)

    #Loss Calculation
    if pair_selection == 'random':
        if_same, euclidean_dist = utils.rand_pairs(results, labels)
    elif pair_selection == 'knn':
        if_same, euclidean_dist = utils.knn_pairs(results,labels)
    elif pair_selection == 'lsh':
        if_same, euclidean_dist = utils.lsh_pairs(results,labels)

    loss = torch.mean(
            (if_same) * torch.pow(euclidean_dist, 2) +
            (1 - if_same) * torch.pow(torch.clamp(loss_margin - euclidean_dist, min=0), 2))
    
    #print("Iteration:", it, "\t\tLoss:", loss.item())
    accumulator += loss.item()
    losses.append(loss.item())

    #Backward Pass
    loss.backward()

    #print(w1.grad)
    
    with torch.no_grad():
        # Update Gradients
        w1 -= learn_rate * w1.grad
        w2 -= learn_rate * w2.grad
        w3 -= learn_rate * w3.grad
        w4 -= learn_rate * w4.grad
        w5 -= learn_rate * w5.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        w4.grad.zero_()
        w5.grad.zero_()

    if it % 100 == 0:
        print("Iteration:", it, "\t|\tAverage Loss (last 100):", round(accumulator/100))
        accumulator = 0

    if it % (iterations/2) == 0:
        utils.graph_points(results, labels, it)

utils.graph_loss(losses)
