import AI
import data
import coherent_networks
import numpy as np
import copy
import os
import pickle
import matplotlib.pyplot as plt

if os.path.exists("con.pickle"):
    with open("con.pickle","rb") as f:
        losses = pickle.load(f)
else:
    dataset = data.load_balanced_emnist()
    data.process_entropy(dataset[0],dataset[1])
    data.process_pca(dataset[0],dataset[1],num=200)
    rng = np.random.default_rng(seed=64572351282468234)
    length = 19
    layout = [200]+[50]*length+[47]
    cons = [None, coherent_networks.connectivity(length+2,1,[3])
            ,coherent_networks.connectivity(length+2,2,[]),  coherent_networks.connectivity(length+2,3,[4])]

    losses = [0,0,0,0]

    for i in range(0,3):
        x = AI.network(layout, seed=rng.integers(2**63), method = "YamAdam")
        x2 = copy.deepcopy(x)
        x3 = copy.deepcopy(x)
        x4 = copy.deepcopy(x)
        x2.convert_incoherent(cons[1],sparseMethod = "GRigL", sparsity = 0.10)
        x3.convert_incoherent(cons[2],sparseMethod = "GRigL", sparsity = 0.10)
        x4.convert_incoherent(cons[3],sparseMethod = "GRigL", sparsity = 0.10)
        if i == 0:
            losses[0] = np.array(x.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
            losses[1] = np.array(x2.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
            losses[2] = np.array(x3.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
            losses[3] = np.array(x4.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
        else:
            losses[0] = losses[0]*(i/(i+1)) + np.array(x.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)
            losses[1] = losses[1]*(i/(i+1)) + np.array(x2.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)
            losses[2] = losses[2]*(i/(i+1)) + np.array(x3.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)
            losses[3] = losses[3]*(i/(i+1)) + np.array(x4.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)

    with open("con.pickle","wb") as f:
        pickle.dump(losses,f)

with open("res.pickle","rb") as f:
    loss = pickle.load(f)[2]

fig, axs = plt.subplots(1, 2)
x = range(0,20)
axs[1].plot(x, losses[0][0],label = 'coherent')
axs[1].plot(x, losses[1][0],label = 'ResNetX')
axs[1].plot(x, losses[2][0],label = 'MaxDist1')
axs[1].plot(x, losses[3][0],label = 'MaxDist2')
axs[1].plot(x, loss[0],label = 'identity_fixed')

axs[0].plot(x, losses[0][1],label = 'coherent')
axs[0].plot(x, losses[1][1],label = 'ResNetX')
axs[0].plot(x, losses[2][1],label = 'MaxDist1')
axs[0].plot(x, losses[3][1],label = 'MaxDist2')
axs[0].plot(x, loss[1],label = 'identity_fixed')

axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

axs[1].legend()
axs[0].legend()
plt.show()
