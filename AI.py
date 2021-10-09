import numpy as np

def activation(x):
    return 1/(1+np.exp(-x))
def load_mnist():
    f=open("./MNIST/train-images.idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(60000,784)/255.0;f.close()
    f=open("./MNIST/train-labels.idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./MNIST/t10k-images.idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(10000,784)/255.0;f.close()
    f=open("./MNIST/t10k-labels.idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    return [images, labels, [784,10,60000,10000]], [test_images, test_labels]

def load_balanced_emnist():
    f=open("./EMNIST/Balanced/emnist-balanced-train-images-idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(112800,784)/255.0;f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-train-labels-idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-images-idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(18800,784)/255.0;f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-labels-idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    return [images, labels, [784,62,112800,18800]], [test_images, test_labels]

def backprop(train,test,layout):
    np.random.seed(42)
    layout = [train[2][0]] + layout + [train[2][1]]
    #layout = [784,10,10]
    BATCH = 4
    ETA = 100
    LAMBDA = 0.001
    ALPHA = 1 - 0.01
    W = []
    nodes= [0]
    dnodes = [0]
    for i in range(0,len(layout)-1):
        W.append(np.random.random((layout[i+1],layout[i])))
        nodes.append(0)
        dnodes.append(0)
    dW = [np.zeros(i.shape) for i in W]

    for a in range(0,100):
        randints = np.random.randint(train[2][2],size=BATCH*10000)
        for i in range(0,10000):
            #forward pass:
            x = randints[i*BATCH:(i+1)*BATCH]
            nodes[0] = np.transpose(train[0][x])
            correct = train[1][x]
            for n,i in enumerate(W):
                nodes[n+1] = activation(LAMBDA*i.dot(nodes[n]))

            target = np.zeros(layout[-1]*BATCH)
            target[np.arange(0,BATCH*layout[-1],layout[-1])+correct]=1
            target = np.transpose(target.reshape(BATCH,layout[-1]))

       #Back propagation:
            dnodes[-1] = 2*(nodes[-1]-target)*ETA*nodes[-1]*(1-nodes[-1])*LAMBDA
            for i in range(len(W)-1,-1,-1):
                dW[i] *= ALPHA
                dW[i] += np.einsum('ji,ki->jk',dnodes[i+1], nodes[i])
                dnodes[i] = np.transpose(W[i]).dot(dnodes[i+1])*nodes[i]*(1-nodes[i])*LAMBDA
                W[i] -= dW[i]

        nodes[0] = np.transpose(test[0])
        for n,i in enumerate(W):
            nodes[n+1] = activation(LAMBDA*i.dot(nodes[n]))
        target = np.zeros(np.size(nodes[-1]))
        target[np.arange(0,np.size(test[0],0)*np.size(nodes[-1],0),np.size(nodes[-1],0))+test[1]]=1
        target = np.transpose(target.reshape(np.size(test[1]),layout[-1]))
        c = np.sum(np.argmax(nodes[-1],0)==test[1])
        print(c, c/train[2][3],np.sum((nodes[-1]-target)**2))

        
class network:
    def __init__(self,layout, blank=False):
        self.layout = layout
        self.input = layout[0]
        self.output = layout[-1]
        self.weights=[]
        self.nodes=[0]*(len(layout))
        self.fit = 0
        self.correct = 0
        self.size = 0
        for i in range(1,len(layout)):
            if not blank:
                self.weights.append(np.random.random((layout[i],layout[i-1])))
            self.size+=layout[i]*layout[i-1]
    def activation(self,x):
        return 1/(1+np.exp(x))
    
    def forward(self,data):
        self.nodes[0] = data
        for i in range(1,len(self.nodes)):
            self.nodes[i] = self.activation(self.weights[i-1].dot(self.nodes[i-1]))
        return self.nodes[-1]

    def pos(self,x):
        n=0
        while x>=np.size(self.weights[n]):
            x-=np.size(self.weights[n])
            n+=1
        return n, x

    def fitness(self,data,labels):
        p = self.forward(data)
        targets = np.zeros(np.size(p))
        targets[np.arange(0,np.size(data,1)*self.output,self.output)+labels]=1
        targets = np.transpose(targets.reshape(np.size(data,1),self.output))
        self.fit=np.sum((targets-p)*(targets-p))
        self.correct = np.sum(labels==np.argmax(p,0))
        return self.fit
            
def genetic(pop,generations,data,labels,layout):
    BATCH = 1024
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
    
        
