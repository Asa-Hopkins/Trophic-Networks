import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Dsigmoid(x,y):
    return x*(1-x)

def tanh(x):
    return np.tanh(x)

def Dtanh(x,y):
    return (1-x**2)

def load_mnist():
    f=open("./MNIST/train-images.idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(60000,784)/255.0;f.close()
    f=open("./MNIST/train-labels.idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./MNIST/t10k-images.idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(10000,784)/255.0;f.close()
    f=open("./MNIST/t10k-labels.idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    return [images, labels, [784,10,60000]], [test_images, test_labels,[784,10,10000]]

def load_balanced_emnist():
    f=open("./EMNIST/Balanced/emnist-balanced-train-images-idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(112800,784)/255.0;f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-train-labels-idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-images-idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(18800,784)/255.0;f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-labels-idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    return [images, labels, [784,62,112800]], [test_images, test_labels,[784,62,18800]]

class network:
    def __init__(self,layout,activation,BATCH=16,seed=42,ETA=0.00001,ALPHA = 0.99):
        self.layout = layout
        self.ETA = ETA
        self.BATCH = BATCH
        self.seed = seed
        self.ALPHA = ALPHA
        self.best = [[1e99,0],[0,0]] #Best fitness in [0], most correct in [1], and which iteration it's achieved.
        self.activation = activation #activation[0] is activation function, [1] is its derivative.
        np.random.seed(seed)
        self.W = []
        self.nodes = [0]
        self.dnodes = [0]
        self.pnodes = [0] #Contains value of nodes before activation function, used for some derivatives
        for i in range(0,len(layout)-1):
            self.W.append(np.random.randn(layout[i+1],layout[i])*np.sqrt(1/layout[i]))
            self.nodes.append(0)
            self.dnodes.append(0)
            self.pnodes.append(0)
        self.dW = [np.zeros(i.shape) for i in self.W]

    def train(self, data, test_data,EPOCHS=10):
        for a in range(EPOCHS):
            randints = np.random.randint(data[2][2],size=(data[2][2] - data[2][2]%self.BATCH)) #train for EPOCHS, testing after each
            for i in range(int(data[2][2]/self.BATCH)):
                x = randints[i*self.BATCH:(i+1)*self.BATCH]
                self.nodes[0] = np.transpose(data[0][x])
                correct = data[1][x]
                #Forward pass
                for n,i in enumerate(self.W):
                    self.pnodes[n+1] = i.dot(self.nodes[n])
                    #print(self.nodes[n+1])
                    self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
                    
                target = np.zeros(self.layout[-1]*self.BATCH)
                target[np.arange(0,self.BATCH*self.layout[-1],self.layout[-1])+correct]=1
                target = np.transpose(target.reshape(self.BATCH,self.layout[-1]))
                                    
                #Back propagation:
                self.dnodes[-1] = 2*(self.nodes[-1]-target)*self.ETA*self.activation[1](self.nodes[-1],self.pnodes[-1])
                for i in range(len(self.W)-1,-1,-1):
                    self.dW[i] *= self.ALPHA
                    self.dW[i] += np.einsum('ji,ki->jk',self.dnodes[i+1], self.nodes[i])
                    self.dnodes[i] = np.transpose(self.W[i]).dot(self.dnodes[i+1])*self.activation[1](self.nodes[i],self.pnodes[i])
                    self.W[i] -= self.dW[i]
                
            self.nodes[0] = np.transpose(test_data[0])
            for n,i in enumerate(self.W):
                self.pnodes[n+1] = i.dot(self.nodes[n])
                self.nodes[n+1] = self.activation[0](self.pnodes[n+1])
            target = np.zeros(np.size(self.nodes[-1]))
            target[np.arange(0,test_data[2][2]*np.size(self.nodes[-1],0),np.size(self.nodes[-1],0))+test_data[1]]=1
            target = np.transpose(target.reshape(np.size(test_data[1]),self.layout[-1]))
            c = np.sum(np.argmax(self.nodes[-1],0)==test[1])/test_data[2][2]
            loss = np.sum((self.nodes[-1]-target)**2)
            if loss < self.best[0][0]:
                self.best[0] = [loss,a]
            if c > self.best[1][0]:
                self.best[1] = [c,a]
            print(loss, c)
           
def genetic(pop,generations,data,labels,layout):
    #BATCH = [2**x for x in range(8)]
    #ETA = [

    
    x = np.random.randint(60000,size=BATCH)
    In = np.transpose(data[x]/255.0)
    correct = labels[x]
    def selection(networks):
        networks = sorted(networks, key=lambda net: net.fit,reverse=False)
        networks = networks[:int(0.2 * len(networks))]
        return networks
    
    def crossover(networks,pop):
        while len(networks)<pop:
            choices = list(range(0,len(networks)))
            parent1 = networks[choices.pop(np.random.randint(len(choices)))]
            parent2 = networks[choices.pop(np.random.randint(len(choices)))]
            child1 = network(parent1.layout,blank=True)
            child2 = network(parent1.layout,blank=True)
            s = np.random.randint(parent1.size)
            for i in parent1.weights:
                child1.weights.append(np.copy(i))
                child2.weights.append(np.copy(i))
                
            networks.append(child1)
            networks.append(child2)
            continue
        
            n, s = child1.pos(s)
            y = (np.arange(0,np.size(child1.weights[n]))<s).reshape(child1.weights[n].shape)
            for i in range(len(child1.weights)):
                if i<n:
                    child1.weights[i]=np.copy(parent2.weights[i])
                elif i>n:
                    child2.weights[i]=np.copy(parent2.weights[i])
            child1.weights[n]=np.copy(parent2.weights[n]*y+parent1.weights[n]*(1-y))
            child2.weights[n]=np.copy(parent1.weights[n]*y+parent2.weights[n]*(1-y))
            networks.append(child1)
            networks.append(child2)
        return networks
        
    def mutation(networks):
        for i in networks:
            for n in range(len(i.weights)):
                y = (np.random.random(i.weights[n].shape)<0.01)
                i.weights[n] = i.weights[n]*(1-y) + np.random.random(i.weights[n].shape)*y
        return networks
    
    
    networks = []
    for i in range(pop):
        networks.append(network(layout))
        networks[i].fitness(In,correct)
    for i in range(generations):
        networks = selection(networks)
        print([i.correct for i in networks])
        print("Generation ", i, ": Best fitness",networks[0].fit, " with", networks[0].correct, " correct predictions")
        networks = crossover(networks,pop)
        networks = mutation(networks)
        for n in networks[int(0.2 * len(networks)):]:
            n.fitness(In,correct)        
