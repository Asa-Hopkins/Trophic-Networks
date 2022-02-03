import numpy as np
import coherent_networks
import copy
from scipy import sparse
import time
import iteround #Has methods for rounding whilst preserving sum
#train, test = load_mnist(); x = network([784,10,10], sparseMethod="SWD", sparsity=[0.1,1e-3,1]); x.train(train,test, EPOCHS=30);
def sigmoid(x):
    return 1/(1+np.exp(-x))

def Dsigmoid(x,y): #y is value of node before activation function is applied
    return x*(1-x)

def tanh(x):
    return np.tanh(x)

def Dtanh(x,y):
    return (1-x*x)

def load_mnist():
    """Loads the classic MNIST data set, returns a pair of lists (one for the training set, one for the testing set) containing
    the images,labels, inputs per image, number of classes and number of images."""
    f=open("./MNIST/train-images.idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(60000,784)/255.0;f.close()
    f=open("./MNIST/train-labels.idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./MNIST/t10k-images.idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(10000,784)/255.0;f.close()
    f=open("./MNIST/t10k-labels.idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    images -= np.mean(images); test_images -= np.mean(test_images)
    return [images, labels, 784, 10, 60000], [test_images, test_labels, 784, 10, 10000]

def load_balanced_emnist():
    """Loads the extended MNIST data set, returns a pair of lists (one for the training set, one for the testing set) containing
        the images,labels, inputs per image, number of classes and number of images."""
    f=open("./EMNIST/Balanced/emnist-balanced-train-images-idx3-ubyte", 'rb'); images = np.frombuffer(f.read()[16:], dtype=np.uint8).reshape(112800,784)/255.0; f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-train-labels-idx1-ubyte", 'rb'); labels = np.frombuffer(f.read()[8:], dtype=np.uint8); f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-images-idx3-ubyte", 'rb'); test_images = np.frombuffer(f.read()[16:], dtype=np.uint8).reshape(18800,784)/255.0; f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-labels-idx1-ubyte", 'rb'); test_labels = np.frombuffer(f.read()[8:], dtype=np.uint8); f.close()
    return [images, labels, 784, 47, 112800], [test_images, test_labels, 784, 47, 18800]


class network:
    def __init__(self, layout, activation=[lambda x: np.tanh(x), lambda x,y: 1-x*x], BATCH=32, seed=42, method="Momentum", hyper=[0.9,2e-4], sparsity=0, sparseMethod="SNFS", incoherence=False):
        self.fit = 0
        self.size = 0
        self.correct = 0
        self.sparsity = sparsity # List containing hyperparameters epsilon and zeta (from 1707.04780)
        self.sparseMethod = sparseMethod
        self.method = method
        self.hyper = hyper #Contains hyperparameters for the learning method used, e.g momentum and learning rate
        self.layout = np.array(layout)
        if np.any(np.array(incoherence,dtype=object)):
            self.incoherence = [np.array(a) for a in incoherence] #A list of lists containing which layers to connect to
            self.out = [np.sum(self.layout[a]) for a in self.incoherence]
        else:
            self.incoherence = incoherence
            self.out = self.layout[1:] #number of output nodes for each layer
        self.BATCH = BATCH
        self.seed = seed
        self.best = [[1e99, 0], [0,0], 0] #Best fitness in [0], most correct in [1], and which iteration it's achieved.
        self.activation = activation #activation[0] is activation function, [1] is its derivative.
        self.rng = np.random.default_rng(seed=seed)
        self.W = [] #List of weight matrices
        if sparsity:
            self.WT = [] #Transpose of that matrix (speeds up sparse calculations)
            self.dW1 = [] #Dense gradient
            self.rows = [] #Row coordinates of nonzero elements
            self.cols = [] #Column coordinates of nonzero elements
            self.s = np.zeros(len(layout)-1) #Size of each layer
            self.meanMom = np.zeros(len(layout)-1) #Mean momentum for a layer, used for SNFS
            if sparseMethod != "SWD":
                scale = 0
                size = 0
                for n in range(0, len(layout)-1):
                    scale += self.out[n]+self.layout[n]
                    size += self.out[n]*self.layout[n]
                scale = (np.exp(sparsity[0]) - 1)*size / scale
                print(sparsity[0]*size)
        self.nodes = [0]
        self.dnodes = [0]
        self.pnodes = [0] #Contains value of nodes before activation function, used for some derivatives
        for n in range(0, len(layout)-1):
            if sparsity and sparseMethod!="SWD":
                temp = int(scale*(self.out[n]+self.layout[n])) #Use Erdős-Rényi initialisation from Mocanu et al.
                #This is lazy since it doesn't check for duplicates, it just ignores them. Calculation of scale contains a probabilistic correction to this.
                x,y = self.rng.integers(0,self.out[n],temp), self.rng.integers(0,self.layout[n],temp)
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
            self.size += np.size(self.W[-1])
            self.nodes.append(0)
            self.dnodes.append(0)
            self.pnodes.append(0)
        self.dW = [i*0 for i in self.W]
        self.g = copy.deepcopy(self.dW)
        self.rms = copy.deepcopy(self.dW)

    def train(self, data, test_data,EPOCHS=10):
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
                    
                target = np.zeros(self.layout[-1]*self.BATCH)
                target[np.arange(0,self.BATCH*self.layout[-1],self.layout[-1])+correct] = 1
                target = target.reshape(self.BATCH,self.layout[-1]).T
                                 
                #Back propagation:
                if self.method == "Momentum":
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])*self.hyper[1]*(1-2/EPOCHS)**a
                    #includes term for annealing of learning rate, should reach about 10% of initial by the end
                    #Changin self.hyper[1] directly caused issues with reproducibility
                    if self.sparseMethod == "SWD":
                        keep = int(self.size*self.sparsity[0]) #self.sparsity[0] is target sparsity
                        threshold = np.partition(np.hstack([np.abs(i).ravel() for i in self.W]),-keep)[-keep]
                        WD = self.sparsity[1]*(self.sparsity[2]/self.sparsity[1])**((a + (j*self.BATCH)/data[4])/EPOCHS)
                    for n in range(len(self.W)-1,-1,-1):
                        if self.sparsity and self.sparseMethod!="SWD":
                            self.dW[n].data *= self.hyper[0]
                            if (data[4]//self.BATCH - j < 2/self.hyper[0] and self.sparseMethod == "SNFS") or (data[4]//self.BATCH == j+1 and self.sparseMethod == "RigL"):
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
                            self.W[n] -= self.dW[n]
                            if self.sparseMethod == "SWD":
                                temp = self.W[n]*(np.abs(self.W[n]) < threshold)
                                self.W[n] -= temp * WD

            #Measure performance against test set:
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
                self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
            #print(np.sum(np.abs(self.nodes[-1])))
            target = np.zeros(test_data[4]*self.layout[-1])
            target[np.arange(0,test_data[4]*self.layout[-1],self.layout[-1])+test_data[1]]=1
            target = target.reshape(test_data[4],self.layout[-1]).T
            c = np.sum(np.argmax(self.nodes[-1],0)==test_data[1])/test_data[4]
            loss = np.sum((self.nodes[-1]-target)**2)
            if loss < self.best[0][0]:
                self.best[0] = [loss,a]
            if c > self.best[1][0]:
                self.best[1] = [c,a]
            self.best[2] = a
            
            if self.sparsity and self.sparsity[1] and self.sparseMethod!="SWD": #remove self.sparsity[1] edges, introduce that many new ones
                p = self.sparsity[1] * (1- a/EPOCHS) #Linear annealing
                
                if self.sparseMethod == "SNFS":
                    for n in range(len(self.W)):
                        self.s[n] = np.size(self.W[n]) #edges per layer before removal
                        self.meanMom[n] = np.mean(np.abs(self.dW[n].data)); #mean absolute momentum
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

                    elif self.sparseMethod == "RigL":
                        remove = int(self.s[n]*p)
                        temp = np.argpartition(self.W[n].data,remove)[:remove]
                        self.W[n].data[temp] = 0
                        self.W[n].eliminate_zeros()
                        self.W[n] = sparse.lil_matrix(self.W[n])
                        self.dW1[n][self.W[n].nonzero()] = 0
                        if remove!=0:
                            x, y = np.unravel_index(np.argpartition(np.abs(self.dW1[n]),-remove, axis=None)[-remove:], self.dW1[n].shape)
                            self.W[n][x,y] = 1e-10
                            
                    self.WT[n] = sparse.csr_matrix(self.W[n].T)
                    self.W[n] = sparse.csr_matrix(self.W[n])
                    self.dW[n] = copy.deepcopy(self.W[n])
                    self.dW[n].data *= 0
                    self.dW1[n] *= 0
                    self.cols[n], self.rows[n], _ = sparse.find(self.WT[n])
            #    self.hyper[1] *= 1-2/EPOCHS #Decay learning rate, should be around 10% of initial by the end
            print(loss, c, [np.size(i) for i in self.W], np.sum([np.size(i) for i in self.W]))
        return c
