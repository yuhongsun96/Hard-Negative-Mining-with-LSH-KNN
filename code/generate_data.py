import random

num_pts = 10000    #Number of points
data_d = 100       #Dimension of data points

#Generate the points under the data directory
file_name = str(num_pts) + "exp" + str(data_d) + "d.csv"
with open("../data/" + file_name, "w") as f:
    #One point per line
    for line in range(num_pts):
        #Class label is first value
        f.write(str(random.randint(0, 1)))
        #Rest of values represents the values along data_d dimensions
        for dim in range (data_d):
            f.write(", " + str(random.uniform(-1, 1)))
        f.write("\n")
