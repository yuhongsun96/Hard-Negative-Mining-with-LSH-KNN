import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils

# Input file, first column are labels, rest are values for dimensions
data_file = "../data/100exp10d.csv"

# Number of intermediate checkpoints to graph
checkpoints = 6

# cuda for GPU, cpu for CPU
mode = torch.device('cuda')

# Parameters
learn_rate = 1e-4
iterations = 10000
loss_margin = 50            # Margin for Contrastive Loss
momentum = 0.9              # Gradient Descent Momentum
pair_selection = "knn"      # random, knn, or lsh


# Reading Data
print("Preparing Data")
data = []
labels = []
for ind, line in enumerate(open(data_file)):
    labels.append(np.array(int(line[0])))
    data.append(np.array(np.fromstring(line.split(',', 1)[1], sep = ',')))

# Convert data to numpy representation
data = np.column_stack(data)
labels = np.column_stack(labels)

# Change numpy matrices to torch tensors of type Float
labels = torch.tensor(labels.transpose()).to(torch.float)
data = torch.tensor(data.transpose()).to(torch.float)

# Move to GPU if mode is CUDA
if mode == torch.device('cuda'):
    data = data.cuda()
    labels = labels.cuda()

# Create Neural Net
print("Preparing Model")
# Define layer widths
D_in, D_1, D_2, D_3, D_4, D_out = data.shape[1], 100, 50, 25, 10, 3

# Define Model Architecture
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_1),
    torch.nn.ReLU(),
    torch.nn.Linear(D_1, D_2),
    torch.nn.ReLU(),
    torch.nn.Linear(D_2, D_3),
    torch.nn.ReLU(),
    torch.nn.Linear(D_3, D_4),
    torch.nn.ReLU(),
    torch.nn.Linear(D_4, D_out),
)
# Move to GPU if mode is CUDA
if mode == torch.device('cuda'):
    model.cuda()

# Define Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

# Training
print("Commence Learning!!!")
accumulator = 0
losses = []
lsh_hash_ptr = 0
checkpoint_count = 0
easy_learning = 1
for it in range(iterations):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    results = model(data)
    
    # Pair selection
    if easy_learning or pair_selection == 'random':    
        if_same, euclidean_dist = utils.rand_pairs(results, labels)
    elif pair_selection == 'knn':
        if_same, euclidean_dist = utils.knn_pairs(results, labels, it, mode)
    elif pair_selection == 'lsh':
        if_same, euclidean_dist, lsh_hash = utils.lsh_pairs(results, labels, it, lsh_hash_ptr)
        
    # Loss Calculation
    loss = torch.mean((if_same) * torch.pow(euclidean_dist, 2) +
            (1 - if_same) * torch.pow(torch.clamp(loss_margin - euclidean_dist, min=0), 2))
    accumulator += loss.item()

    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print averaged losses to keep track of progress
    if it % 100 == 0:
        if it == 0:
            print("Iteration:", it, "\t|\tInitial Loss:", accumulator)
            losses.append(accumulator)
        else:
            avg_loss = accumulator / 100
            print("Iteration:", it, "\t|\tAverage Loss (last 100):", avg_loss)
            losses.append(avg_loss)
            # If average loss drops below threshold, begin using hard negative mining
            if avg_loss < 10 and easy_learning:
                utils.graph_points(results, labels, "Easy_Learning_Final")
                easy_learning = 0
                if pair_selection != 'random':
                    print("Switch pair selection from random to", pair_selection)
        accumulator = 0
    
    # Create checkpoing graphs to view feature space evolution
    if it % round(iterations / (checkpoints + 1)) == 0:
        utils.graph_points(results, labels, pair_selection + "_" + str(checkpoint_count))
        checkpoint_count += 1

utils.graph_points(results, labels, pair_selection + "_final")
utils.graph_loss(losses, pair_selection)
