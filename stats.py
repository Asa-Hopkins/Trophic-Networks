import csv
import numpy as np
from scipy.stats import shapiro
from scipy.stats import ttest_ind
#Ordering in table is [1,2, 9,10, 3,4, 5,6, 11,12, 15,16, 19,20, 21,22,23,24, 7,8, 13,14, 17,18, 27,25,26]
order = [1,2, 9,10, 3,4, 5,6, 11,12, 15,16, 19,20, 21,22,23,24, 7,8, 13,14, 17,18, 27,25,26]
pairs = [(1,2),(9,10),(3,4),(5,6),(11,12),(15,16),(19,20),(21,22),(21,23),(21,24),(13,14),(17,18),(27,25),(27,26)]
draw = [13,14,17,18,25,27]
with open("tests.csv", newline='') as f:
    reader = csv.reader(f,delimiter=';',quotechar='"')
    results = []
    normal = []
    mean = []
    std = []
    DoM = []
    connect = []
    l = [784,784,200,200] #length of input layers for each dataset.
    dataset = []
    for row in reader:
        if row[0] == "ID":
            continue
        results.append([float(row[10+i]) for i in range(0,20)])
        mean.append(np.mean(results[-1]))
        std.append(np.std(results[-1], ddof=1))
        coherence = [float(row[32+i]) for i in range(0,20)]
        normal.append(shapiro(results[-1])[1])
        dataset.append(int(row[1]))
        a = [ [l[int(row[1])]] + eval(row[8]),eval(row[9])]
        connect.append(a)
        L_B = a[0][0] * sum([a[0][i] for i in a[1][0]])# * float(row[6])
        L = 0
        for b in range(0,len(a[0])-1):
            L += a[0][b] * sum([a[0][b+i] for i in a[1][b]])# * float(row[6])
        if np.mean(coherence)!=0:
            print(shapiro(coherence)[1] < 0.05, normal[-1] < 0.05)
            #print(row[7])
            #print(coherence)
            #print(row[0], np.sqrt(L/L_B - 1),np.mean(coherence))
            pass
    for a,b in pairs:
        a -= 1
        b -= 1
        print((a+1,b+1), ttest_ind(results[a], results[b], equal_var = False)[1])
        
    for a in draw: #create .dot entry for this network
        continue #comment out if needed
        with open(str(a)+".dot","w") as g:
            a-=1
            g.write("digraph g {\n")
            c = connect[a]
            c[1] = c[1] + [[]]
            nodes = ["Input"] + [f"Hidden{i+1}" for i in range(len(c[0])-2)] + ["Output"]
            for i in range(len(c[0])):
                w = 5*np.sqrt(c[0][i]/max(c[0]))
                g.write(f"{nodes[i]} [width = {w}]\n")
                for b in c[1][i]:
                    g.write(f"{nodes[i]} -> {nodes[i+b]};\n")
            g.write("}")
                
