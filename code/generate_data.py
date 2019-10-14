import random

num_pts = 100   #Number of points
data_d = 10     #Dimension of data points

#Generate the points under the data directory
with open("../data/data.csv", "w") as f:
    #One point per line
    for line in range(num_pts):
        #Class label is first value
        f.write(str(random.randint(0, 1)))
        #Rest of values represents the values along data_d dimensions
        for dim in range (data_d):
            f.write(", " + str(random.uniform(-1, 1)))
        f.write("\n")
