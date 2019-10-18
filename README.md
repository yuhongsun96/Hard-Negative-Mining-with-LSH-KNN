# Hard-Negative-Mining-with-LSH-KNN
A demo of Hard Negative Mining using a simple feed forward neural net with contrastive loss. Results compared between baseline random pair selection, GPU KNN and GPU LSH KNN. Network outputs a 3D feature space for visualization purposes.

**To run the demo either view the jupyter notebook file TrainingNotebook.ipynb or execute python3 train.py**

If the libraries knn_lib.so or lsh_lib.so do not work for your platform, compile it from:
https://github.com/yuhongsun96/Python-GPU-KNN
https://github.com/yuhongsun96/Python-GPU-LSH

By the default the code runs in cuda mode, if cuda is not available, change: mode = torch.device('cpu')

### Main features:
- The neural network is a fully connect feedforward network with 5 layers with relu as activation aside from the output layer.
- The input dimensions can vary allowing for more flexible uses.
- The output is 3D for visualizing the changes in the feature space as training progresses.
- Since the data used for the demo is randomly generated with no actual underlying pattern, there is no training/test split.
  The purpose is to demonstrate the benefits of different example selection methods.
  
### Results (10/18/19):
- Using random pairing, the model is able to clearly separate the data. 
  The inherent randomness of this pairing allows it to overcome areas with low gradients.
  With equivalent number of iterations to KNN, the groups are more spread out.
- With KNN, the model creates tighter clusters with larger margins in general.
  This approach sometimes faces slow learning at certain stages. Since the largest loss pairs are selected every iteration, there is not much randomness between the iterations.
  Introducing batching or momentum may fix this. 
- LSH is potentially facing some hardware limitations. Only a small number of points can be hashed before the library breaks.
  Without being able to hash a good portion of the reference points, accurate results are not possible. More testing is required.
