import AI
import data
import optuna
import csv
import coherent_networks
import numpy as np

datasets = [data.load_mnist(),data.load_balanced_emnist(),
            data.load_mnist(),data.load_balanced_emnist()]
data.process_entropy(datasets[2][0],datasets[2][1])
data.process_pca(datasets[2][0],datasets[2][1],num=200)

data.process_entropy(datasets[3][0],datasets[3][1])
data.process_pca(datasets[3][0],datasets[3][1],num=200)
rng = np.random.default_rng(seed=64572351282468234)

def dense_objective(trial, layout, ds):
    hyper = [trial.suggest_float("hyper0",0.85,0.95),
             trial.suggest_float("hyper1",1e-5,1e-3),
             trial.suggest_float("hyper2",0.9,0.98)]
    x = AI.network(layout, seed=rng.integers(2**63), hyper = hyper)
    return -x.train(datasets[ds][0], datasets[ds][1], EPOCHS = 50)

def sparse_objective(trial, layout, connectivity, ds, sparsity, EPOCHS):
    hyper = [trial.suggest_float("hyper0",0.85,0.95),
             trial.suggest_float("hyper1",1e-5,1e-3),
             trial.suggest_float("hyper2",0.9,0.98)]
    sparsity = [sparsity,1e-3,1]
    x = AI.network(layout, seed=rng.integers(2**63), hyper=hyper, sparseMethod = "SWD", sparsity = sparsity, incoherence = connectivity)
    return -x.train(datasets[ds][0], datasets[ds][1], EPOCHS = EPOCHS)

with open("tests.csv", newline='') as f:
    g = open("tests2.csv", mode="w",newline='')
    writer = csv.writer(g,delimiter = ";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    reader = csv.reader(f,delimiter=';',quotechar='"')
    for row in reader:
        runs = []
        q = []
        study = optuna.create_study()
        if row[0] == "ID" or row[12]!="":
            [rng.integers(2**63) for a in range(0,40)] # ensure reproducibility
            writer.writerow(row)
            continue
        ds = int(row[1])
        
        if row[2] == "0": #Traditional architecture
            exec("layout =" + row[8])
            layout = np.array([datasets[ds][0][2]] + layout)
            dense_edges = sum(layout[1:]*layout[:-1])
            exec("connectivity ="+row[9])
            if row[6] == "1": #If dense
                edges = dense_edges
                sparsity = 0
                #[rng.integers(2**63) for a in range(0,40)]
                #continue
                study.optimize(lambda trial: dense_objective(trial, layout, ds),
                               n_trials = 20)
                temp = study.best_params
                hyper = [temp["hyper0"],temp["hyper1"],temp["hyper2"]]
                for i in range(0,20):
                    x = AI.network(layout, seed=rng.integers(2**63), hyper=hyper)
                    runs.append(x.train(datasets[ds][0], datasets[ds][1], EPOCHS = 50))
                    q.append(0)
            else:
                if row[6] !="":
                    sparsity = float(row[6])
                else:
                    raise Exception
                edges = dense_edges * sparsity
                study.optimize(lambda trial: sparse_objective(trial, layout, connectivity,
                                                              ds, sparsity, 50), n_trials = 20)
                temp = study.best_params
                hyper = [temp["hyper0"],temp["hyper1"],temp["hyper2"]]
                for i in range(0,20):
                    x = AI.network(layout, seed=rng.integers(2**63), hyper=hyper,
                                   sparseMethod = "SWD", sparsity = [sparsity,1e-3,1],
                                   incoherence = connectivity)
                    runs.append(x.train(datasets[ds][0], datasets[ds][1], EPOCHS = EPOCHS))

        if row[2] == "1": #New architecture
            exec("layout =" + row[8])
            layout = np.array([datasets[ds][0][2]] + layout)
            exec("connectivity ="+row[9])
            dense_edges2 = 0
            for n,i in enumerate(layout[:-1]):
                dense_edges2 += sum([i*layout[n+c] for c in connectivity[n]])
            if sparsity == 0:
                sparsity = 1
            sparsity = sparsity * dense_edges/dense_edges2
            EPOCHS = int(50 * dense_edges/dense_edges2)
            print(edges, dense_edges2,sparsity, EPOCHS)
            study.optimize(lambda trial: sparse_objective(trial, layout, connectivity,
                                                              ds, sparsity, EPOCHS), n_trials = 20)
            edges = 0
            temp = study.best_params
            hyper = [temp["hyper0"],temp["hyper1"],temp["hyper2"]]
            for i in range(0,20):
                temp = 1e99
                while temp > 1.05*dense_edges:
                    x = AI.network(layout, seed=rng.integers(2**63), hyper=hyper,
                                   sparseMethod = "SWD", sparsity = [sparsity,1e-3,1],
                                   incoherence = connectivity)
                    temp2 = x.train(datasets[ds][0], datasets[ds][1], EPOCHS = EPOCHS)
                    x.convert()
                    temp = sum([a.size for a in x.W])
                runs.append(temp2)
                q.append(coherent_networks.analyse(x))
                edges += sum([a.size for a in x.W])/20
        writer.writerow(row[0:3] + hyper + [sparsity if sparsity!=0 else 1] + [edges] + row[8:10] + runs + [np.mean(runs)] + [np.std(runs)] + q + [np.mean(q)] + [np.std(q)])
    g.close()
