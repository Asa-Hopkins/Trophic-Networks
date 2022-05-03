import numpy as np
import coherent_networks
import copy
from scipy import sparse
import iteround #Has methods for rounding whilst preserving sum
import data
#train, test = data.load_mnist();x = network([train[2],10,10]); x.train(train,test, EPOCHS=10);
#train, test = data.load_mnist();x = network([train[2],10,10],method="YamAdam"); x.train(train,test, EPOCHS=10);
#train, test = data.load_mnist();data.process_entropy(train,test);data.process_pca(train,test,num=200);

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Dsigmoid(x,y): #y is value of node before activation function is applied
    return x*(1-x)

def tanh(x):
    return np.tanh(x)

def Dtanh(x,y):
    return (1 - x*x)


class network:
    def __init__(self, layout, activation=[tanh, Dtanh], BATCH=64, seed=42, method="Momentum", hyper=[0.9,2e-4, 0.9], sparsity=0, sparseMethod="SNFS", incoherence=False, mask = False):
        self.fit = 0
        self.size = 0
        self.correct = 0
        self.sparsity = sparsity # List containing hyperparameters epsilon and zeta (from 1707.04780)
        self.msk = mask # Whether to use sparse matrices or dense matrices with masking
        self.sparseMethod = sparseMethod
        self.method = method
        self.hyper = hyper #Contains hyperparameters for the learning method used, e.g momentum and learning rate
        if self.method == "vSGD":
            self.hyper = [1,1,1]
        else:
            self.eta = self.hyper[1] #Eta will attenuate over training, but it's important that the original value is kept
        self.layout = np.array(layout)
        if np.any(np.array(incoherence,dtype=object)):
            self.incoherence = [np.array(a) for a in incoherence] #A list of lists containing which layers to connect to
            self.out = [np.sum(self.layout[n+a]) for n,a in enumerate(self.incoherence)]
        else:
            self.incoherence = incoherence
            self.out = self.layout[1:] #number of output nodes for each layer
        self.BATCH = BATCH
        self.seed = seed
        self.best = [[1e99, 0], [0,0], 0] #Best fitness in [0], most correct in [1], and which iteration it's achieved.
        self.activation = activation #activation[0] is activation function, [1] is its derivative.
        self.rng = np.random.default_rng(seed=seed)
        self.W = [] #List of weight matrices
        self.s = [] #Size of each layer
        
        if sparsity:
            self.mask = []
            self.WT = [] #Transpose of that matrix (speeds up sparse calculations)
            self.dW1 = [] #Dense gradient
            self.rows = [] #Row coordinates of nonzero elements
            self.cols = [] #Column coordinates of nonzero elements
            self.meanMom = np.zeros(len(layout)-1) #Mean momentum for a layer, used for SNFS
            if sparseMethod != "SWD":
                scale = 0
                size = 0
                for n in range(0, len(layout)-1):
                    scale += self.out[n]+self.layout[n]
                    size += self.out[n]*self.layout[n]
                scale = np.log(1-sparsity[0])/np.log(1-1/size) / scale
        self.nodes = [0]
        self.dnodes = [0]
        self.ddnodes = [0]
        self.pnodes = [0] #Contains value of nodes before activation function, used for some derivatives
        for n in range(0, len(layout)-1):
            if sparsity and sparseMethod!="SWD":
                temp = int(scale*(self.out[n]+self.layout[n])) #Use Erdős-Rényi initialisation from Mocanu et al.
                #This is lazy since it doesn't check for duplicates, it just ignores them. Calculation of scale contains a probabilistic correction to this.
                x,y = self.rng.integers(0,self.out[n],temp), self.rng.integers(0,self.layout[n],temp)
                if self.msk:
                    temp2 = np.zeros((self.out[n],self.layout[n]))
                    temp2[x,y] = 1
                    self.mask.append(copy.deepcopy(temp2))
                    temp2 *= self.rng.standard_normal(temp2.shape) * np.sqrt(self.out[n]/temp)
                    self.W.append(temp2)
                else:
                    temp = sparse.coo_matrix( (self.rng.standard_normal(temp) * np.sqrt(self.out[n]/temp), [x,y]),(self.out[n],self.layout[n]))
                    self.W.append(temp.tocsr())
                    self.dW1.append(np.zeros(np.shape(temp)))
                    self.WT.append(self.W[-1].T)
                temp = np.nonzero(self.W[-1])
                self.rows.append(temp[0])
                self.cols.append(temp[1])
            else:
                if self.incoherence:
                    #For incoherent networks, matrix describes all nodes that this layer connects to
                    #By doing this, backwards propagation is hardly changed, but forwards propagation is more complicated.
                    #It's possible to instead leave forwards propagation unchanged, but this seemed the better option
                    self.W.append(self.rng.standard_normal((self.out[n],self.layout[n])) * np.sqrt(1/(self.layout[n]*len(self.incoherence[n]))))
                else:
                    self.W.append(self.rng.standard_normal((self.out[n],self.layout[n])) * np.sqrt(1/(self.layout[n])))
            if type(self.W[n])!=type(np.array([1,2,3])):
                self.s.append(self.W[n].size)
            else:
                self.s.append(np.count_nonzero(self.W[n]))
            self.size += self.s[-1]
            self.nodes.append(0)
            self.dnodes.append(0)
            self.pnodes.append(0)
        self.dW = [i*0 for i in self.W]
        if self.method=="Adam":
            self.m = copy.deepcopy(self.dW) # average gradient
            self.v = copy.deepcopy(self.dW) # average gradient squared
        if self.method=="YamAdam":
            self.m = copy.deepcopy(self.dW) # average gradient
            self.v = copy.deepcopy(self.dW) # average gradient squared
            self.su = copy.deepcopy(self.dW) 
            self.h = copy.deepcopy(self.dW) 
            self.h2 = [0]*len(self.dW)
            self.h1 = [0]*len(self.dW)
            self.b = [0]*len(self.dW)

    def train(self, data, test_data, EPOCHS=10, patience = 4):
        if self.method == "Momentum":
            self.eta = self.hyper[1]
        if self.method == "Adam":
            t = 0
        history = [[],[]]
        trigger = 0
        for a in range(EPOCHS):
            randints = self.rng.integers(data[4],size=(data[4] - data[4]%self.BATCH)) #train for EPOCHS, testing after each
            for j in range(data[4]//self.BATCH):
                x = randints[j*self.BATCH:(j+1)*self.BATCH]
                self.nodes[0] = data[0][x].T
                correct = data[1][x]
                
                for i in range(len(self.W)):
                    self.pnodes[i+1] = np.zeros((self.layout[i+1],self.BATCH)) #set to correct shapes
                    self.nodes[i+1] = np.zeros((self.layout[i+1],self.BATCH))
                #Forward pass
                for n,i in enumerate(self.W):
                    if self.incoherence:
                        temp = i.dot(self.nodes[n]) #calculate contribution to all future layers
                        for k in self.incoherence[n]:
                            self.pnodes[n+k] += temp[:len(self.pnodes[n+k])]
                            temp = temp[len(self.pnodes[n+k]):]
                    else:
                        self.pnodes[n+1] = i.dot(self.nodes[n])
                    self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
                    
                target = np.zeros(self.layout[-1]*self.BATCH)# - 1
                target[np.arange(0,self.BATCH*self.layout[-1],self.layout[-1])+correct] = 1
                target = target.reshape(self.BATCH,self.layout[-1]).T

                #Back propagation:
                if self.method == "Adam":
                    t += 1
                    alpha = self.hyper[0]
                    beta1 = 0.9
                    beta2 = 0.999
                    epsilon = 1e-6
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])#/self.BATCH
                    for n in range(len(self.W)-1,-1,-1):
                        self.dW[n] = self.dnodes[n+1].dot(self.nodes[n].T)
                        self.m[n] += (self.dW[n] - self.m[n])*(1-beta1)
                        self.v[n] += (self.dW[n]**2 - self.v[n])*(1-beta2)
                        self.W[n] -= alpha*self.m[n]/(1-beta1**t)/(np.sqrt(self.v[n]/(1-beta2**t))+epsilon)
                            
                        self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])

                if self.method == "YamAdam":
                    epsilon = 1e-6
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])#/self.BATCH
                    for n in range(len(self.W)-1,-1,-1):
                        if self.incoherence:
                            dnodes = np.vstack([self.dnodes[n+k] for k in self.incoherence[n]])
                        else:
                            dnodes = self.dnodes[n+1]
                                
                        self.dW[n] = dnodes.dot(self.nodes[n].T)
                        temp = 1-self.b[n]
                        self.v[n] *= self.b[n]
                        self.v[n] += temp*(self.dW[n] - self.m[n])**2
                        self.m[n] *= self.b[n]
                        self.m[n] += temp*self.dW[n]
                        self.su[n] *= self.b[n]
                        self.su[n] += temp*self.h[n]**2
                        self.h2[n] = self.h1[n]
                        if self.msk:
                            self.h[n] = np.sqrt((self.su[n] + epsilon)/(self.v[n] + epsilon))*self.m[n]*self.mask[n]
                        else:
                            self.h[n] = np.sqrt((self.su[n] + epsilon)/(self.v[n] + epsilon))*self.m[n]
                        self.h1[n] = np.sum(np.abs(self.h[n])) + epsilon
                        self.b[n] = 1 / (1 + np.exp(-self.h2[n]/self.h1[n])) - epsilon
                        self.W[n] -= self.h[n]
                        self.dnodes[n] = (self.W[n].T).dot(dnodes)*self.activation[1](self.nodes[n],self.pnodes[n])
                
                if self.method == "Momentum":
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])*self.eta
                    for n in range(len(self.W)-1,-1,-1):
                        if self.sparsity and self.sparseMethod!="SWD" and self.msk == False:
                            self.dW[n].data *= self.hyper[0]
                            if (data[4]//self.BATCH - j < 2/self.hyper[0] and self.sparseMethod == "SNFS") or (data[4]//self.BATCH - j < 2/self.hyper[0] and self.sparseMethod == "RigL"):
                                self.dW1[n] *= self.hyper[0]
                            if self.incoherence:
                                temp = np.vstack([self.dnodes[n+k] for k in self.incoherence[n]])
                                self.dW1[n] += temp.dot(self.nodes[n].T)
                                self.dW[n].data += np.einsum('ik,ik->i',temp.take(self.rows[n],0),self.nodes[n].take(self.cols[n],0))
                                self.dnodes[n] = (self.WT[n]).dot(temp)*self.activation[1](self.nodes[n],self.pnodes[n])
                            else:
                                self.dW1[n] += self.dnodes[n+1].dot(self.nodes[n].T)
                                self.dW[n].data += np.einsum('ik,ik->i',self.dnodes[n+1].take(self.rows[n],0),self.nodes[n].take(self.cols[n],0))
                                self.dnodes[n] = (self.WT[n]).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            self.W[n].data -= self.dW[n].data
                        else:
                            self.dW[n] *= self.hyper[0]
                            if self.incoherence:
                                temp = np.vstack([self.dnodes[n+k] for k in self.incoherence[n]])
                                self.dW[n] += temp.dot(self.nodes[n].T)
                                self.dnodes[n] = (self.W[n].T).dot(temp)*self.activation[1](self.nodes[n],self.pnodes[n])
                            else:    
                                self.dW[n] += self.dnodes[n+1].dot(self.nodes[n].T)
                                self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            if self.msk:
                                self.W[n] -= self.dW[n]*self.mask[n]
                            else:
                                self.W[n] -= self.dW[n]
                                
                if self.sparseMethod == "SWD":
                    keep = int(self.size*self.sparsity[0]) #self.sparsity[0] is target sparsity
                    threshold = np.partition(np.hstack([np.abs(i).ravel() for i in self.W]),-keep)[-keep]
                    WD = self.sparsity[1]*(self.sparsity[2]/self.sparsity[1])**((a + (j*self.BATCH)/data[4])/EPOCHS)
                    for n in range(len(self.W)-1,-1,-1):
                        temp = self.W[n]*(np.abs(self.W[n]) < threshold)
                        self.W[n] -= temp * WD
                        
            #Measure performance against test set:
            c, loss = self.test(test_data)
            history[0].append(c)
            history[1].append(loss)
            #Early stopping
            if loss < self.best[0][0]:
                self.best[0] = [loss,a]
                trigger = 0
            else:
                trigger += 1
            if c > self.best[1][0]:
                self.best[1] = [c,a]
            self.best[2] = a
            if trigger == patience:
                print(loss, c, [np.size(i) for i in self.W], np.sum([np.size(i) for i in self.W]))
                return [c,loss,a,history]
            
            if self.sparsity and self.sparsity[1] and self.sparseMethod!="SWD": #remove self.sparsity[1] edges, introduce that many new ones
                p = self.sparsity[1] * (1- a/EPOCHS) #Linear annealing
                
                if self.sparseMethod == "SNFS":
                    for n in range(len(self.W)):
                        self.s[n] = np.size(self.W[n]) #edges per layer before removal
                        self.meanMom[n] = np.mean(np.abs(self.dW[n].data)) #mean absolute momentum

                    self.meanMom/=np.sum(self.meanMom)
                    totalRemove = 0
                    for n in range(len(self.W)):
                        p = np.min((p,1-self.s[n]/(self.layout[n]*self.out[n])))
                        remove = int(p*self.s[n])
                        if remove==0:
                            self.dW1[n][self.W[n].nonzero()] = 0
                            continue
                        temp = np.argpartition(np.abs(self.W[n].data),remove)[:remove]
                        self.W[n].data[temp] = 0
                        self.W[n].eliminate_zeros()
                        totalRemove+=remove
                        self.dW1[n][self.dW1[n] == 0] = 1e-10
                        self.dW1[n][self.W[n].nonzero()] = 0 #This lets us avoid picking already nonzero elements
                        #Skipping this line accidentally causes permanent removal of edges
                    add = totalRemove*self.meanMom
                    
                    excess = add - (self.out*self.layout[:-1]) + self.s# get excess edges
                    while np.any(excess>0):
                        excess[np.where(excess < 0)]=-np.sum(excess[np.where(excess>0)])/np.size(excess[np.where(excess < 0)])
                        add -= excess
                        excess = add - np.array([np.sum(self.dW1[n]!=0) for n in range(len(self.W))]) #get excess edges
                        # this covers an edge case where some nodes have no connections, giving dW1 = 0 for some edges
                        # Not an ideal solution, as allowing these edges to be made could be benificial.
                    add = np.array(iteround.saferound(add,0), dtype=np.int32)
                    for n in range(len(self.W)):
                        if add[n]==0:
                            continue
                        temp = np.argpartition(np.abs(self.dW1[n].ravel()),-add[n])[-add[n]:]
                        x, y = np.unravel_index(temp,self.W[n].shape)
                        self.W[n] = sparse.lil_matrix(self.W[n])
                        self.W[n][x,y] = 1e-20
                    
                for n in range(len(self.W)):
                    
                    if self.sparseMethod == "Random":
                        matrix_data = np.concatenate([i.data for i in self.W])
                        remove = int(self.sparsity[1]*self.size)
                        temp = np.argpartition(np.abs(matrix_data),remove)[:remove]
                        matrix_data[temp] = 0 #remove lowest few values of all edges (not per layer)
                        self.W[n].data = matrix_data[:np.size(self.W[n])]
                        matrix_data = matrix_data[np.size(self.W[n]):]
                        self.W[n].eliminate_zeros()
                        self.W[n] = sparse.lil_matrix(self.W[n])
                        self.s[n] = np.size(self.W[n]) #remaining edges per layer
                        while self.s[n]>0: #Need to be careful about picking already nonzero elements when choosing new edges to add
                            x, y = self.rng.integers(0,self.out[n]), self.rng.integers(0,self.layout[n]) #not an ideal solution but shouldn't be a huge bottleneck
                            if self.W[n][x,y] == 0:
                                self.W[n][x,y] = 1e-10
                                self.s[n]-=1

                    if self.sparseMethod == "RigL":

                        remove = int(self.s[n]*p)
                        if remove == 0:
                            continue
                        if self.msk == False:
                            temp = np.argpartition(np.abs(self.W[n].data),remove)[:remove]
                            self.W[n].data[temp] = 0
                            self.W[n].eliminate_zeros()
                            self.W[n] = sparse.lil_matrix(self.W[n])
                            self.dW1[n][self.dW1[n] == 0] = 1e-10
                            self.dW1[n][self.W[n].nonzero()] = 0
                            x, y = np.unravel_index(np.argpartition(np.abs(self.dW1[n]),-remove, axis=None)[-remove:], self.dW1[n].shape)
                            self.W[n][x,y] = 1e-10
                        else:
                            nonzeros = np.nonzero(self.W[n])
                            smallest = np.argpartition(np.abs(self.W[n][nonzeros]),remove,axis=None)[:remove]
                            self.mask[n][nonzeros[0][smallest],nonzeros[1][smallest]] = 0
                            self.W[n][nonzeros[0][smallest],nonzeros[1][smallest]] = 0
                            x, y = np.unravel_index(np.argpartition(np.abs(self.dW[n]*(1-self.mask[n])) + 1e-10*(1-self.mask[n]),-remove, axis=None)[-remove:], self.dW[n].shape)
                            self.mask[n][x,y] = 1
                            
                    if self.msk ==False:        
                        self.WT[n] = sparse.csr_matrix(self.W[n].T)
                        self.W[n] = sparse.csr_matrix(self.W[n])
                        self.dW[n] = copy.deepcopy(self.W[n])
                        self.dW[n].data *= 0
                        self.dW1[n] *= 0
                        self.cols[n], self.rows[n], _ = sparse.find(self.WT[n])

                                            
                if self.sparseMethod == "GRigL":
                    edges = int(self.size*p)
                    flat = np.concatenate([i.flat for i in self.W])
                    nonzeros = np.nonzero(flat)
                    remove = np.sort(nonzeros[0][np.argpartition(np.abs(flat[nonzeros]),edges)[:edges]])
                    for n in range(len(self.W)):
                        size = self.W[n].size
                        temp = np.searchsorted(remove,size)
                        if temp == 0:
                            remove -= size
                            continue
                        indices = remove[:temp]
                        flat[indices] = 0
                        self.W[n][np.unravel_index(indices,self.W[n].shape)] = 0
                        self.mask[n][np.unravel_index(indices,self.W[n].shape)] = 0
                        remove = remove[temp:]
                        remove -= size
                    zeros = np.where(flat==0)
                    flat = np.concatenate([i.flat for i in self.dW])
                    add = np.sort(zeros[0][np.argpartition(np.abs(flat[zeros]),-edges)[-edges:]])
                    for n in range(len(self.W)):
                        size = self.W[n].size
                        temp = np.searchsorted(add,size)
                        if temp == 0:
                            add -= size
                            continue
                        indices = add[:temp]
                        self.mask[n][np.unravel_index(indices,self.W[n].shape)] = 1
                        add = add[temp:]
                        add -= size
                
            if self.method == "Momentum":
                self.eta = self.hyper[1]/(a+1) #Annealing of learning rate
            print(loss, c, [np.size(i) for i in self.W], np.sum([np.size(i) for i in self.W]))
        return [c,loss,a,history]

    def test(self,test_data, off = {}):
        self.nodes[0] = test_data[0].T
        for i in range(len(self.W)):
            self.pnodes[i+1] = np.zeros((self.layout[i+1],test_data[4])) #set to correct shapes
            self.nodes[i+1] = np.zeros((self.layout[i+1],test_data[4]))
        
        for n,i in enumerate(self.W):
            if self.incoherence:
                temp = i.dot(self.nodes[n]) #calculate contribution to all future layers
                for k in self.incoherence[n]:
                    self.pnodes[n+k] += temp[:len(self.pnodes[n+k])]
                    temp = temp[len(self.pnodes[n+k]):]
            else:
                self.pnodes[n+1] = i.dot(self.nodes[n])
            if n+1 in off.keys():
                self.pnodes[n+1][off[n+1]] *= 0
            self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
        target = np.zeros(test_data[4]*self.layout[-1])# - 1
        target[np.arange(0,test_data[4]*self.layout[-1],self.layout[-1])+test_data[1]] = 1
        target = target.reshape(test_data[4],self.layout[-1]).T
        c = np.sum(np.argmax(self.nodes[-1],0)==test_data[1])/test_data[4]
        loss = np.sum((self.nodes[-1]-target)**2)
        return c, loss
    
    def convert(self,sparseMethod = "SNFS", sparsity = 0.1, threshold = 1e-6, mask=True): #converts a SWD network to a true sparse network, and allows for sparse training
        self.msk = mask
        sparsity = [0,sparsity]
        self.size = 0
        self.s = []
        if mask:
            self.mask = []
            self.dW = []
        else:
            self.WT = []
            self.dW = []
            self.dW1 = []
            self.cols = []; self.rows = []
        if sparseMethod == "SNFS":
            self.meanMom = np.zeros(len(self.layout)-1) #Mean momentum for a layer, used for SNFS
        if sparseMethod == "RigL" or sparseMethod == "GRigL" :
            self.s = []
            for i in self.W:
                self.s.append(np.count_nonzero(i))
        for n,i in enumerate(self.W):
            if mask:
                sparsity[0] += np.sum([np.abs(i) > threshold])
                i[np.abs(i) < threshold] = 0
                self.mask.append(copy.deepcopy(i!=0))
                self.dW.append(copy.deepcopy(i))
                self.dW[-1] *= 0
                self.W[n] = i
            else:
                self.dW1.append(i*0)
                sparsity[0] += np.sum([np.abs(i) > threshold])
                i[np.abs(i) < threshold] = 0
                i = sparse.coo_matrix(i)
                self.WT.append(sparse.csr_matrix(i.T))
                i = sparse.csr_matrix(i)
                self.dW.append(copy.deepcopy(i))
                self.dW[-1].data *= 0
                a, b, _ = sparse.find(self.WT[-1])
                self.cols.append(a)
                self.rows.append(b)
                self.W[n] = i
            self.s.append(np.count_nonzero(self.W[n]))
            self.size+=self.W[n].size
        if self.method == "YamAdam":
            self.m = copy.deepcopy(self.dW) # average gradient
            self.v = copy.deepcopy(self.dW) # average gradient squared
            self.su = copy.deepcopy(self.dW) 
            self.h = copy.deepcopy(self.dW) 
            self.h2 = [0]*len(self.dW)
            self.h1 = [0]*len(self.dW)
            self.b = [0]*len(self.dW)
        self.sparseMethod = sparseMethod
        sparsity[0] /= self.size
        self.sparsity = sparsity
        
    def convert_incoherent(self,incoherence, sparseMethod = "SNFS", sparsity = 0.1, diagonal = 0): #Converts a coherent network to a (sparse) incoherent one.
        #The idea here is to convert to dense incoherent with the new edges all 0, so the conversion to sparse gets the right amount of edges.
        self.incoherence = incoherence
        self.incoherence = [np.array(a) for a in incoherence] #A list of lists containing which layers to connect to
        self.out = [np.sum(self.layout[n+a]) for n,a in enumerate(self.incoherence)]
        for n in range(len(self.W)):
            i = self.W[n].shape
            self.W[n] = np.vstack((self.W[n],diagonal*np.eye(self.out[n]-i[0],i[1])/np.sqrt(self.layout[n]) ))
        self.convert(sparseMethod = sparseMethod, sparsity = sparsity, mask=True)
        
