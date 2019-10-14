import torch
import matplotlib.pyplot as plt

def rand_pairs(data, labels):
    perm = torch.randperm(data.size(0))
    equals = torch.eq(labels, labels[perm]).squeeze().to(torch.float)
    dists = torch.sum((data - data[perm]) ** 2, 1)
    return equals, dists
    

def graph_loss(losses):
    plt.plot(range(len(losses))[100:], losses[100:])
    plt.savefig('loss_over_training.png')

def graph_points(points, labels, iteration):
    points = points.cpu().detach().numpy()
    print(points.shape)
    x1 = points[:, 0]
    x2 = points[:, 1]
    x3 = points[:, 2]
    print(x1)
    print(x2)
    print(x3)
    y = labels.cpu().detach().numpy()
    plt.scatter(x1, x2,x3)
    plt.show()

