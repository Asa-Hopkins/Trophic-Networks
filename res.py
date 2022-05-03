import AI
import data
import coherent_networks
import numpy as np
import copy
import os
import pickle
import matplotlib.pyplot as plt

if os.path.exists("res.pickle"):
    with open("res.pickle","rb") as f:
        losses = pickle.load(f)
        with open("con.pickle","rb") as f:
            losses[0] = pickle.load(f)[-2]
        
else:
    dataset = data.load_balanced_emnist()
    data.process_entropy(dataset[0],dataset[1])
    data.process_pca(dataset[0],dataset[1],num=200)
    rng = np.random.default_rng(seed=64572351282468234)
    length = 19
    layout = [200]+[50]*length+[47]
    cons = [None, coherent_networks.connectivity(length+2,3,[4])]

    losses = [0,0,0,0]
    if os.path.exists("con.pickle"):
        with open("con.pickle","rb") as f:
            losses[0] = pickle.load(f)[-1]

    for i in range(0,5):
        x = AI.network(layout, seed=rng.integers(2**63), method = "YamAdam")
        x2 = copy.deepcopy(x)
        x3 = copy.deepcopy(x)
        x4 = copy.deepcopy(x)
        x2.convert_incoherent(cons[1],sparseMethod = "GRigL", sparsity = 0.1, diagonal = 1)
        x3.convert_incoherent(cons[1],sparseMethod = "GRigL", sparsity = 0, diagonal = 1)
        x4.convert_incoherent(cons[1],sparseMethod = "GRigL", sparsity = 0.25, diagonal = 1)
        if i == 0:
            losses[1] = np.array(x2.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
            losses[2] = np.array(x3.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
            losses[3] = np.array(x4.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])
        else:
            losses[1] = losses[1]*(i/(i+1)) + np.array(x2.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)
            losses[2] = losses[2]*(i/(i+1)) + np.array(x3.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)
            losses[3] = losses[3]*(i/(i+1)) + np.array(x4.train(dataset[0],dataset[1],EPOCHS = 20, patience = 50)[3])/(i+1)

    with open("res.pickle","rb") as f:
        pickle.dump(losses,f)

fig, axs = plt.subplots(1, 2)
x = range(0,20)
axs[0].plot(x, losses[0][1],label = 'coherent_start')
axs[0].plot(x, losses[1][1],label = 'Identity_start')
axs[0].plot(x, losses[2][1],label = 'Identity_fixed')
axs[0].plot(x, losses[3][1],label = 'Identity_start2')

axs[1].plot(x, losses[0][0],label = 'coherent_start')
axs[1].plot(x, losses[1][0],label = 'Identity_start')
axs[1].plot(x, losses[2][0],label = 'Identity_fixed')
axs[1].plot(x, losses[3][0],label = 'Identity_start2')

axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

axs[1].legend()
axs[0].legend()
plt.show()
