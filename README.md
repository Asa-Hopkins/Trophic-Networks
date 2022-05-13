# Trophic-Networks
An experiment on how neural network incoherence affects network accuracy and training speed. This is the topic of my fourth year dissertation and so full details will be accessible there.

## Usage
The file `AI.py` provides a class for creating simple neural networks with a range of network layouts, learning methods and sparsity structures. The innovation is to allow a new parameter called the connectivity, which describes which layers connections can be made between. Typically, nodes of one layer only connect to the next layer, but now there can be an array of layers which it can make connections to. To sum it up briefly, the advantage of doing this is to leverage the fast training speed of shallow networks, whilst still achieving the accuracy of deep networks.

## Similar work
Similar work has been done by [ResNet](https://arxiv.org/pdf/1505.00387) and [ResNetX](https://arxiv.org/abs/1912.12165), which in context of this project can be interpreted as a fixed set of connectivity arrays and a fixed set of connections within those arrays. The difference here is that any connectivity can be used, and the choice of active connections is handled by a sparse training algorithm. This allows the network incoherence to vary naturally during training, rather than being a fixed hyperparameter like in these papers.

## Results
In the graphs below, a network with layout `[200] + [50]*18 + [47]` is trained on the EMNIST balanced image database, after undergoing dimensionality reduction using PCA to bring it down to 200 inputs. The networks each have an equal number of nodes, with the coherent network being dense. Training is done with YamAdam, a hyperparameter-free variant of Adam, and sparse training is done by a modified version of RigL.
The modifications are to allow RigL to remove edges from one layer, and to add them back in another, as opposed to only placing edges in the same layer as they were taken from.
A horizontal line is drawn where each network hits 74% accuracy, to give an indication of convergence speed.

In the first test, a range of connectivities are tried, using the fully coherent network as a starting point for the sparse training algorithm to build on. This shows that any incoherence is greatly helpful to network training, both for final accuracy and speed of convergence, but also that the particular choice of connectivity is important too. All to All gives the highest accuracy as expected, but also exhibits slow initial convergence which indicaes the training method starting to struggle.

![](https://github.com/Asa-Hopkins/Trophic-Networks/blob/main/Results/Results1.png)

In the second test, MaxDist2 is used with `t = 3` for the relay layer frequency. Here, a range of starting edge configurations are used. Here, 'identity' means ResNet's suggested configuration. In the 'fixed' test, the edges are not moved during training, whereas in the other test they are allowed to be moved by the sparse training method.
This shows that a good choice of starting configuration can effectively skip an epoch, but it is more important to allow sparse network training as it is unlikely this starting configuartion is optimal. Similarly, a bad choice of starting configuration can completely ruin the network, so care must be taken.

![](https://github.com/Asa-Hopkins/Trophic-Networks/blob/main/Results/Results2.png)

## Conclusions
Confirming ResNet's results, it appears that for particularly deep networks it is advantageous for the network to have some incoherence.
It turns out however that ResNet's choice of connectivity and active connections aren't optimal, and some more optimal connecitivities are suggested here. These can be seen at the bottom of the file `coherent_networks.py`. 
The idea behind these are to assert a maximum number of jumps necessary to reach the output of the network from a given node.
Doing this directly tackles the issue of a node being too deep to train effectively, as there will now be a path requiring just one or two jumps, for example. 
These connectivities are called "MaxDistN", where N is the maximum number of jumps being asserted. Beyond MaxDist1, there is a great deal of possible choices that satisfy the requirements of MaxDistN, and so the method used here is only one possible method. MaxDist2 uses relay layers, so each layer connects to the next relay layer, and then the relay layer connects to the output layer.
The frequency of relay layers is left a hyperparameter
This then propagates back to the weights themselves. In a sense, error is a nonincreasing function of the number of allowed connections, since the new connections can remain unused without affecting the current minimum. This means the expected best connectivity is an all to all connectivity, and this is confirmed in tests.

The issue with this becomes the learning method, as this creates a larger search space that must be covered. For this reason it is still useful to have more restricted connectivity models. This gives three new models, "MaxDist1", "MaxDist2" and "All to All". It is possible to create further MaxDist algorithms, using relays of relays and so on, but this not only introduces more hyperparameters but also captures even fewer of the possibilities that still satisfy MaxDistN.

## Future work
Some more immediate future developments are to adapt this work for use on convolutional networks and recurrent (e.g LSTM) networks. This would allow a more direct comparison to ResNet, and would also be much more useful as it's rare to see simple neural networks in use these days.

It would also be useful to have a more systematic way of picking a connectivity within the MaxDist classes, rather than just leaving hyperparameters.
