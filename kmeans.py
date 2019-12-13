'''
Run kmeans analysis on real dataset 

'''

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import axes3d, Axes3D  
import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.metrics import silhouette_samples, silhouette_score

#load dataset from Mantz et al. 2015 paper 
def get_data(file_name):
    data = pd.read_csv(file_name, sep='&', header=None)
    symmetry = data[4].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    symmetry_err = data[6].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    peakiness = data[7].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    peakiness_err = data[9].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    alignment = data[10].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    alignment_err = data[12].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    ellipticity = data[14].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    ellipticity_err = data[16].str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False).astype('float64').get_values()
    label = data[3]
    name = data[1]
    return name, symmetry, symmetry_err, peakiness, peakiness_err, alignment, alignment_err, ellipticity, ellipticity_err, label



name, symmetry, symmetry_err, peakiness, peakiness_err, alignment, alignment_err, ellipticity, ellipticity_err, tag = get_data('giant_table_all.tex') 


sym = symmetry[~np.isnan(symmetry)] 
names = name[~np.isnan(alignment)]
names = np.array(names) 
peak = peakiness[~np.isnan(alignment)]
align = alignment[~np.isnan(alignment)]

#Clusters in A08 Sample 
mask4 = np.zeros(len(tag))
for i in range(len(tag)):
    mask4[i] = 'a' in tag[i]
mask4 = mask4[~np.isnan(alignment)]

X = np.array(list(zip(sym, align,peak))) 

inertia = [] 
for i, k in enumerate([2,3,4,5,6,7]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    inertia = np.append(inertia, kmeans.inertia_) 

    print(centroids)
    print(kmeans.inertia_)
    print(kmeans.n_iter_) 
    silhouette_vals = silhouette_samples(X, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(X[:, 0], X[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([0, 2])
    ax2.set_xlim([0, 2])
    ax2.set_xlabel('Symmetry')
    ax2.set_ylabel('Alignment')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',
                 fontsize=16, fontweight='semibold', y=1.05);

    plt.show() 

#elbow plot 
plt.rcParams['figure.figsize'] = (16, 9)
plt.scatter([2,3,4,5,6,7], inertia) 
plt.show() 

#run kmeans analysis using 100 centroid initializations 
kmeans = KMeans(n_clusters=3, n_init=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

#Clusters that were labelled as relaxed by previous SPA analysis 
relax = np.zeros(len(align))
for i in range(len(align)):
    if(align[i] > 1.0 and peak[i] > -.82 and sym[i] > 0.87):
        relax[i] = 1 

mask = labels == labels 
mask2 = relax == 1 
mask4 = mask4.astype(bool) 

labels = (labels + 1)%3 
print(labels)

#generate results plots 
fig, ax = plt.subplots()
ax.scatter(sym[mask], align[mask], marker='o', c=labels[mask]+1, s=150, edgecolors='white')
ax.scatter(sym[mask4], align[mask4], marker='v', c='red', s=75) 
ax.scatter(sym[mask2], align[mask2], marker='x', c='green', s=75)

ax.set_xlabel('symmetry', fontsize=20)
ax.set_ylabel('alignment', fontsize=20)
ax.axvline(x=0.87, linestyle='--')
ax.axhline(y=1.0, linestyle='--')

plt.show()


fig, ax = plt.subplots()
ax.scatter(peak[mask], align[mask], marker = 'o', c=labels[mask], s=150,edgecolors='white') 
ax.scatter(peak[mask4], align[mask4], marker='v', c='red', s=75)
ax.scatter(peak[mask2], align[mask2], marker='x', c='green', s=75)

ax.set_xlabel('peakiness', fontsize=20)
ax.set_ylabel('alignment', fontsize=20)
ax.axvline(x=-0.82, linestyle='--')
ax.axhline(y=1.0, linestyle='--')

plt.show()

fig, ax = plt.subplots()
ax.scatter(peak[mask], sym[mask], marker='o', c=labels[mask], s=150, edgecolors='white')
ax.scatter(peak[mask4], sym[mask4], marker='v', c='red', s=75) 
ax.scatter(peak[mask2], sym[mask2], marker='x', c='green', s=75)

ax.set_xlabel('peakiness', fontsize=20)
ax.set_ylabel('symmetry', fontsize=20)
ax.axvline(x=-0.82, linestyle='--')
ax.axhline(y=0.87, linestyle='--')

plt.show()
