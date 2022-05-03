import AI
import data
import csv
import coherent_networks
import numpy as np
import copy

datasets = [data.load_mnist(),data.load_balanced_emnist(),
            data.load_mnist(),data.load_balanced_emnist(),
            data.load_fmnist()]
data.process_entropy(datasets[2][0],datasets[2][1])
data.process_pca(datasets[2][0],datasets[2][1],num=200)

data.process_entropy(datasets[3][0],datasets[3][1])
data.process_pca(datasets[3][0],datasets[3][1],num=200)

data.process_entropy(datasets[4][0],datasets[4][1])
data.process_pca(datasets[4][0],datasets[4][1],num=200)
rng = np.random.default_rng(seed=64572351282468234)

with open("moretests.csv", newline='') as f:
    g = open("moretests2.csv", mode="w",newline='')
    writer = csv.writer(g,delimiter = ";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    reader = list(csv.reader(f,delimiter=';',quotechar='"'))
    for n,row in enumerate(reader):
        acc = [[]]
        loss = [[]]
        q = [[]]
        base = []
        if row[0] == "ID" or row[6]!="":
            [rng.integers(2**63) for a in range(0,6)] # ensure reproducibility
            writer.writerow(row)
            continue
        ds = int(row[1])
        
        if row[2] == "0": #Traditional architecture
            exec("layout =" + row[4])
            layout = np.array([datasets[ds][0][2]] + layout + [datasets[ds][0][3]])
            exec("connectivity ="+row[5])
            for i in range(0,10):
                x = AI.network(layout, seed=rng.integers(2**63), method = "YamAdam")
                out = x.train(datasets[ds][0], datasets[ds][1], EPOCHS = 50)
                acc[0].append(out[0])
                loss[0].append(out[1])
                l=1
                base.append(x.layout[0]*x.layout[1])
                while len(reader)>n+l and reader[n+l][2] == "1":
                    if len(acc) == l:
                        acc.append([])
                        loss.append([])
                        q.append([])
                    y = copy.deepcopy(x)
                    y.best = [[1e99, 0], [0,0], 0]
                    exec("connectivity ="+reader[n+l][5])
                    y.convert_incoherent(connectivity,sparseMethod = "GRigL", sparsity = 0.03)
                    out = y.train(datasets[ds][0], datasets[ds][1], EPOCHS = 10, patience = 10)
                    acc[l].append(out[0])
                    loss[l].append(out[1])
                    q[l].append(coherent_networks.analyse(y))
                    base.append(np.count_nonzero(y.W[0]))
                    l += 1
                q[0].append(0)
            for i in range(0,l):
                writer.writerow(reader[n+i][0:3] + [base[l]] + reader[n+i][4:6] + acc[i] + [np.mean(acc[i])] + [np.std(acc[i],ddof=1)] + q[i] + [np.mean(q[i])] + [np.std(q[i],ddof=1)] + loss[i] + [np.mean(loss[i])] + [np.std(loss[i],ddof=1)])
        else:continue
    g.close()
