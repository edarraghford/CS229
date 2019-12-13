import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

symmetry = np.loadtxt('symmetry.txt') 
alignment = np.loadtxt('alignment.txt') 
peakiness = np.loadtxt('peakiness.txt') 

res = np.loadtxt('relaxation_values.txt') 
m = res == 1 
print(len(res[m])) 

obs = np.loadtxt('simulation_obs.txt', dtype=str)
shuffled = np.loadtxt('obs_shuffled.txt', dtype=str)
test_set = shuffled[250:]  

obs = pd.Series(obs)
test_set = pd.Series(test_set)

mask = obs.isin(test_set) 

sym = symmetry[mask]
align = alignment[mask]
peak = peakiness[mask]
relax = res[mask]
results1 = np.zeros(len(test_set))
results2 = np.zeros(len(test_set))
obs = obs[mask] 

for i in range(len(test_set)):
    if (align[i] > 1.0 and sym[i] > 0.87):
        results1[i] = 1 
    if (align[i] > 1.0 and sym[i] > 0.87 and peak[i] > -0.82):
        results2[i] = 1 

plt.scatter(sym, align, c=results1)
plt.scatter(sym[relax==1], align[relax==1], color='red', marker = '*') 
#plt.show()

mask = relax == 1 
print(len(relax[mask])) 
cr = np.sum(results1[relax==1])
cur = len(results1[relax==0]) - np.sum(results1[relax==0])
pr = np.round((cr/len(results1[relax==1])), 3) 
pur = np.round((cur/len(results1[relax==0])), 3) 
m = np.zeros(len(relax))
for i in range(len(relax)):
    if(relax[i] == 0 and results1[i] == 1):
        m[i] = 1
print(obs[m==1]) 
print("number correct relaxed: " + str(cr) + "/" + str(len(results1[relax==1])) + " (" + str(pr) + ")" )
print("number correct unrelaxed: " + str(cur) + "/" + str(len(results1[relax==0])) + " (" + str(pur) + ")" )


