'''
Run kmeans analysis on simulated dataset

'''
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import axes3d, Axes3D  
import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.metrics import silhouette_samples, silhouette_score

# load simulated dataset 
symmetry = np.loadtxt('symmetry.txt')
alignment = np.loadtxt('alignment.txt')
peakiness = np.loadtxt('peakiness.txt')
relaxation = np.loadtxt('relaxation_values.txt') 
relaxation_spa = np.loadtxt('relaxation4_values.txt') 

sym = symmetry[~np.isnan(symmetry)]
peak = peakiness[~np.isnan(alignment)]
align = alignment[~np.isnan(alignment)]
relax = relaxation[~np.isnan(alignment)]
relaxspa = relaxation_spa[~np.isnan(alignment)]

X = np.array(list(zip(sym, align, peak)))  #, ellipt)))

# run kmeans analysis 
kmeans = KMeans(n_clusters=3, n_init=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
mask = labels == labels  
mask2 = relaxspa == 1 
mask3 = relax == 1 
print(len(sym))
print(len(sym[mask3]))
print(len(sym[mask2]))

labels = (labels + 2)%3 
print(labels)


#generate plots 

fig, ax = plt.subplots()
ax.scatter(sym[mask], align[mask], marker='o', c=labels[mask], s=120, edgecolors='white') #,ecolor='lightgray', elinewidth=3)
ax.scatter(sym[mask3], align[mask3], marker='v', c='red', s=60) #,ecolor='lightgray', elinewidth=3)
ax.scatter(sym[mask2], align[mask2], marker='x', c='green', s=60) #,ecolor='lightgray', elinewidth=3)
#for i in range(len(names)):
#       ax.annotate(names[i], (sym[i]+0.001, align[i]+0.001), c ='r', size=6, weight='bold')

ax.set_xlabel('symmetry', fontsize=20)
ax.set_ylabel('alignment', fontsize=20)
ax.axvline(x=0.87, linestyle='--')
ax.axhline(y=1.0, linestyle='--')

plt.show()


fig, ax = plt.subplots()
ax.scatter(peak[mask], align[mask], marker = 'o', c=labels[mask], s=120,edgecolors='white') #, elinewidth=3)
ax.scatter(peak[mask3], align[mask3], marker='v', c='red', s=60) #,ecolor='lightgray', elinewidth=3)
ax.scatter(peak[mask2], align[mask2], marker='x', c='green', s=60) #,ecolor='lightgray', elinewidth=3)

ax.set_xlabel('peakiness', fontsize=20)
ax.set_ylabel('alignment',fontsize=20)
ax.axvline(x=-0.82, linestyle='--')
ax.axhline(y=1.0, linestyle='--')

plt.show()

fig, ax = plt.subplots()
ax.scatter(peak[mask], sym[mask], marker='o', c=labels[mask], s=120, edgecolors='white') # ,ecolor='lightgray', elinewidth=3)
ax.scatter(peak[mask3], sym[mask3], marker='v', c='red', s=60) #,ecolor='lightgray', elinewidth=3)
ax.scatter(peak[mask2], sym[mask2], marker='x', c='green', s=60) #,ecolor='lightgray', elinewidth=3)

ax.set_xlabel('peakiness', fontsize=20)
ax.set_ylabel('symmetry', fontsize=20)
ax.axvline(x=-0.82, linestyle='--')
ax.axhline(y=0.87, linestyle='--')

plt.show()
