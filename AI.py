import numpy as np
import coherent_networks
import copy
from scipy import sparse
import iteround #Has methods for rounding whilst preserving sum

#import time; train, test = load_mnist(); x = network([784,10,10],[tanh,Dtanh]); t=time.time(); x.train(train,test); print(time.time()-t)

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
    def __init__(self, layout, activation=[lambda x: np.tanh(x), lambda x,y: 1-x*x], BATCH=32, seed=42, method="Momentum", hyper=[0.99,0.00001], sparsity=0):
        self.fit = 0
        self.size = 0
        self.correct = 0
        self.sparsity = sparsity # List containing hyperparameters epsilon and zeta (from 1707.04780)
        self.method = method
        self.hyper = hyper #Contains hyperparameters for the learning method used, e.g momentum and learning rate
        self.layout = layout
        self.BATCH = BATCH
        self.seed = seed
        self.best = [[1e99, 0], [0,0], 0] #Best fitness in [0], most correct in [1], and which iteration it's achieved.
        self.activation = activation #activation[0] is activation function, [1] is its derivative.
        np.random.seed(seed)
        self.W = [] #List of weight matrices
        self.rows = []
        self.cols = []
        self.nodes = [0]
        self.dnodes = [0]
        self.pnodes = [0] #Contains value of nodes before activation function, used for some derivatives
        for n in range(0, len(layout)-1):
            if sparsity:
                temp = (1-sparsity[0])*(layout[n+1]+layout[n])/(layout[n+1]*layout[n])
                self.W.append(np.random.randn(layout[n+1],layout[n]) * np.sqrt(1/(layout[n]*temp)))
                temp = np.random.random((layout[n+1],layout[n])) < temp
                self.W[-1] = sparse.csr_matrix(self.W[-1]*temp)
                print(np.size(self.W[-1]))
                temp = np.nonzero(temp)
                self.rows.append(temp[0])
                self.cols.append(temp[1])
            else:
                self.W.append(np.random.randn(layout[n+1],layout[n]) * np.sqrt(1/(layout[n])))
            self.size += np.size(self.W[-1])
            self.nodes.append(0)
            self.dnodes.append(0)
            self.pnodes.append(0)
        self.dW = [i*0 for i in self.W]
        self.g = copy.deepcopy(self.dW)
        print(self.size)
        self.rms = copy.deepcopy(self.dW)

    def train(self, data, test_data,EPOCHS=10):
        for a in range(EPOCHS):
            randints = np.random.randint(data[4],size=(data[4] - data[4]%self.BATCH)) #train for EPOCHS, testing after each
            for j in range(data[4]//self.BATCH):
                x = randints[j*self.BATCH:(j+1)*self.BATCH]
                self.nodes[0] = data[0][x].T
                correct = data[1][x]
                
                #Forward pass
                for n,i in enumerate(self.W):
                    self.pnodes[n+1] = i.dot(self.nodes[n])
                    self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
                    
                target = np.zeros(self.layout[-1]*self.BATCH)
                target[np.arange(0,self.BATCH*self.layout[-1],self.layout[-1])+correct] = 1
                target = target.reshape(self.BATCH,self.layout[-1]).T
                                 
                #Back propagation:
                if self.method == "Momentum":
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])*self.hyper[1]
                    for n in range(len(self.W)-1,-1,-1):
                        if self.sparsity:
                            self.dW[n].data *= self.hyper[0]
                            self.dW[n].data += np.einsum('ik,ik->i',self.dnodes[n+1].take(self.rows[n],0),self.nodes[n].take(self.cols[n],0))
                            self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            self.W[n].data -= self.dW[n].data
                        else:
                            self.dW[n] *= self.hyper[0]
                            self.dW[n] += self.dnodes[n+1].dot(self.nodes[n].T)
                            self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            self.W[n] -= self.dW[n]
                
                elif self.method == "AdaDelta":
                    self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.activation[1](self.nodes[-1],self.pnodes[-1])
                    for n in range(len(self.W)-1,-1,-1):                        
                        if self.sparsity: #By using .data it skips verifying coordinates
                            self.rms[n].data *= self.hyper[0]
                            self.rms[n].data += (1-self.hyper[0])*self.dW[n].data**2
                            self.dW[n].data = np.einsum('ik,ik->i',self.dnodes[n+1][self.rows[n],:],self.nodes[n][self.cols[n],:])              
                            self.g[n].data *= self.hyper[0]
                            self.g[n].data += (1-self.hyper[0])*self.dW[n].data**2
                            self.dW[n].data *= np.sqrt((self.rms[n].data+self.hyper[1])/(self.g[n].data+self.hyper[1]))
                            self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            self.W[n].data -= self.dW[n].data

                        else:
                            self.rms[n] *= self.hyper[0]
                            self.rms[n] += (1-self.hyper[0])*self.dW[n]**2
                            self.dW[n] = self.dnodes[n+1].dot(self.nodes[n].T)
                            self.g[n] *= self.hyper[0]
                            self.g[n] += (1-self.hyper[0])*self.dW[n]**2
                            self.dW[n] *= np.sqrt((self.rms[n]+self.hyper[1])/(self.g[n]+self.hyper[1]))
                            self.dnodes[n] = (self.W[n].T).dot(self.dnodes[n+1])*self.activation[1](self.nodes[n],self.pnodes[n])
                            self.W[n] -= self.dW[n]

            #Measure performance against test set:
            self.nodes[0] = test_data[0].T
            for n,i in enumerate(self.W):
                self.nodes[n+1] = self.activation[0](i.dot(self.nodes[n]))
            target = np.zeros(test_data[4]*self.layout[-1])
            target[np.arange(0,test_data[4]*self.layout[-1],self.layout[-1])+test_data[1]]=1
            target = target.reshape(test_data[4],self.layout[-1]).T
            c = np.sum(np.argmax(self.nodes[-1],0)==test[1])/test_data[4]
            loss = np.sum((self.nodes[-1]-target)**2)
            if loss < self.best[0][0]:
                self.best[0] = [loss,a]
            if c > self.best[1][0]:
                self.best[1] = [c,a]
            self.best[2] = a
            if self.sparsity and self.sparsity[1]: #remove self.sparsity[1] edges, introduce that many new ones
                matrix_data = np.concatenate([i.data for i in self.W])
                remove = int(self.sparsity[1]*self.size)
                temp = np.argpartition(np.abs(matrix_data),remove)[:remove]
                matrix_data[temp] = 0 #remove lowest few values of all edges (not per layer)
                s = np.size(self.W[0])
                for n in range(1,len(self.W)):
                    s = np.hstack( (s,np.size(self.W[n])))
                s = iteround.saferound(remove*s/np.sum(s),0)
                for n in range(len(self.W)):
                    self.W[n].data = matrix_data[:np.size(self.W[n])]
                    matrix_data = matrix_data[np.size(self.W[n]):]
                    self.W[n].eliminate_zeros()
                    self.W[n] = sparse.lil_matrix(self.W[n])
                    while s[0]>0: #Need to be careful about picking already nonzero elements when choosing new edges to add
                        x, y = np.random.randint(0,self.layout[n+1]), np.random.randint(0,self.layout[n]) #not an idea solution but shouldn't be a huge bottleneck
                        if self.W[n][x,y] == 0:
                            self.W[n][x,y] = 1e-10#np.random.randn(1) * np.sqrt(self.layout[n+1]/s)
                            s[0]-=1
                    s = np.delete(s,0)
                    self.W[n] = sparse.csr_matrix(self.W[n])
                    self.dW[n] = copy.deepcopy(self.W[n])
                    self.dW[n].data *= 0
                    if self.method == "AdaDelta":
                        self.g[n] = copy.deepcopy(self.dW[n])
                        self.rms[n] = copy.deepcopy(self.dW[n])
                    self.cols[n], self.rows[n], _ = sparse.find(self.W[n].T)
            print(loss, c, np.size(self.W[0]), np.size(self.W[1]))

    def fitness(self,data):
        self.nodes[0] = data[0]
        for n,i in enumerate(self.W):
            self.pnodes[n+1] = i.dot(self.nodes[n])
            self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
        target = np.zeros(self.layout[-1]*data[4])
        target[np.arange(0,data[4]*self.layout[-1],self.layout[-1])+data[1]] = 1
        target = np.transpose(target.reshape(data[4],self.layout[-1]))
        self.fit = np.sum((self.nodes[-1]-target)**2)
        self.correct = np.sum(np.argmax(self.nodes[-1],0)==data[1])
    
           
def genetic(pop,generations,data,test,layout):
    
    x = np.random.randint(60000,size=500)
    In = np.transpose(data[0][x])
    correct = data[1][x]
    def selection(networks):
        networks = sorted(networks, key=lambda net: net.fit,reverse=False)
        networks = networks[:int(0.2 * len(networks))]
        return networks
    
    def crossover(networks,pop):
        while len(networks)<pop:
            choices = list(range(0,len(networks)))
            parent1 = networks[choices.pop(np.random.randint(len(choices)))]
            parent2 = networks[choices.pop(np.random.randint(len(choices)))]
            child1 = network(parent1.layout)
            child2 = network(parent1.layout)
            s = np.random.randint(parent1.size)
            for i in range(len(parent1.W)):
                child1.W[i] = np.copy(parent1.W[i])
                child2.W[i] = np.copy(parent1.W[i])
                
            networks.append(child1)
            networks.append(child2)
            continue
        
            n, s = child1.pos(s)
            y = (np.arange(0,np.size(child1.W[n]))<s).reshape(child1.W[n].shape)
            for i in range(len(child1.W)):
                if i<n:
                    child1.W[i]=np.copy(parent2.W[i])
                elif i>n:
                    child2.W[i]=np.copy(parent2.W[i])
            child1.W[n]=np.copy(parent2.W[n]*y+parent1.W[n]*(1-y))
            child2.W[n]=np.copy(parent1.W[n]*y+parent2.W[n]*(1-y))
            networks.append(child1)
            networks.append(child2)
        return networks
        
    def mutation(networks):
        for i in networks:
            for n in range(len(i.W)):
                y = (np.random.random(i.W[n].shape)<0.05)
                i.W[n] = i.W[n]*(1-y) + np.random.randn(*np.shape(i.W[n]))*np.sqrt(1/i.layout[n])*y
        return networks
    
    
    networks = []
    for i in range(pop):
        networks.append(network(layout))
        networks[i].fitness([In,correct,test[2],test[3],test[4]])
    for i in range(generations):
        networks = selection(networks)
        print("Generation ", i, ": Best fitness",networks[0].fit, " with", networks[0].correct, " correct predictions")
        networks = crossover(networks,pop)
        networks = mutation(networks)
        for n in networks[int(0.2 * len(networks)):]:
            n.fitness([In,correct,test[2],test[3],test[4]])
