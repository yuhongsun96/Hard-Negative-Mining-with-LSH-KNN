import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import knn_py_wrapper as pyKNN
import lsh_py_wrapper as pyLSH

# Select random pairs of points in the final space and return Euclidean distance and if labels matched
def rand_pairs(data, labels):
    perm = torch.randperm(data.size(0))
    equals = torch.eq(labels, labels[perm]).squeeze().to(torch.float)
    dists = torch.sum((data - data[perm]) ** 2, 1)
    return equals, dists


# Use GPU KNN to select hardest examples, furthest points within group, closest points outside of group
def knn_pairs(data, labels, it, mode):
    # Get furthest points of the same group
    # indices where label is 0 or 1
    ind0 = torch.where(labels == 0)[0]
    ind1 = torch.where(labels == 1)[0]
    # data points where label is 0 or 1
    data0 = data[ind0]
    data1 = data[ind1]
    # convert to numpy representation formatted for KNN lib
    ref0 = data0.cpu().detach().numpy().flatten("F")
    ref1 = data1.cpu().detach().numpy().flatten("F")
    ref0_size = int(len(ref0) / 3)
    ref1_size = int(len(ref1) / 3)
    # indices for k-neighbors for sets of points with label 0 or 1
    knn0 = list(pyKNN.gpu_knn(ref0, ref0, 3, ref0_size))
    knn1 = list(pyKNN.gpu_knn(ref1, ref1, 3, ref1_size))
    # indices in the sets of points with label 0 or 1, corresponding to furthest point of that set
    furthest0 = knn0[-ref0_size:]
    furthest1 = knn1[-ref1_size:]
    dists0 = torch.sum((data0 - data0[furthest0]) ** 2, 1)
    dists1 = torch.sum((data1 - data1[furthest1]) ** 2, 1)
    dists_same = torch.cat((dists0, dists1), 0)
    equals_same = torch.ones(labels.shape)
    if mode == torch.device('cuda'):
        equals_same = equals_same.cuda()
        dists_same = dists_same.cuda()

    # Get closest points of the other group
    # indices where label is 0 or 1
    ind0 = torch.where(labels == 0)[0]
    ind1 = torch.where(labels == 1)[0]
    # data points where label is 0 or 1
    data0 = data[ind0]
    data1 = data[ind1]
    # convert to numpy representation formatted for KNN lib
    ref0 = data0.cpu().detach().numpy().flatten("F")
    ref1 = data1.cpu().detach().numpy().flatten("F")
    ref0_size = int(len(ref0) / 3)
    ref1_size = int(len(ref1) / 3)
    # indices for k-neighbors for sets of points with label 0 or 1
    knn0 = list(pyKNN.gpu_knn(ref1, ref0, 3, 2))
    knn1 = list(pyKNN.gpu_knn(ref0, ref1, 3, 2))
    # indices in the sets of points with label 0 or 1, corresponding to furthest point of that set
    furthest0 = knn0[ref0_size:]
    furthest1 = knn1[ref1_size:]
    dists0 = torch.sum((data0 - data1[furthest0]) ** 2, 1)
    dists1 = torch.sum((data1 - data0[furthest1]) ** 2, 1)
    dists_diff = torch.cat((dists0, dists1), 0)
    equals_diff = torch.zeros(labels.shape)
    if mode == torch.device('cuda'):
        equals_diff = equals_diff.cuda()
        dists_diff = dists_diff.cuda()
    return torch.cat((equals_same, equals_diff), 0), torch.cat((dists_same, dists_diff), 0)


# Use GPU LSH to select approximate hard examples
# scales better than KNN since no need to compare every pair
def lsh_pairs(data, labels, it, hash_ptr):
    #TODO complete this function
    raise NotImplementedError
    #LSH parameters
    refresh = 10
    num_tables = 5
    projection_dim = 3
    bucket_wid = 3
    probe_bins = 3
    
    # Prepare Data
    data = data.cpu().detach().numpy().flatten().tolist()[:9]
    
    # Call to LSH lib
    if it % refresh == 0:
        hash_ptr = pyLSH.lsh_hash(hash_ptr, 3, num_tables, projection_dim, bucket_wid, data)
    ind = pyLSH.lsh_search(hash_ptr, 2, probe_bins, len(data), data, data)
    ind = torch.tensor(np.asarray(list(ind)))    
    
    
# Plot and save loss over all iterations
#file name determined by what pair selection method is used
def graph_loss(losses, pair_selection):
    plt.plot(range(0, 100 * len(losses), 100), losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("With " + pair_selection + " pairs")
    plt.ylim(ymin=0)
    plt.savefig('../results/' + pair_selection + '_pairs_loss.png')\


# Visualize Neural Net output in 3D space
def graph_points(points, labels, fname):
    points = points.cpu().detach().numpy()
    x1 = points[:, 0]
    x2 = points[:, 1]
    x3 = points[:, 2]
    y = np.squeeze(labels.cpu().detach().numpy())

    colors = ['red', 'blue']
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, x2, x3, c=y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.savefig('../results/checkpoint_' + fname)
    
