import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score

from sklearn.cluster import Birch, AgglomerativeClustering, KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import f_oneway, ttest_rel

from sknetwork.clustering import Louvain
from sknetwork.topology import  Cliques

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA

import umap

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return js_graph

file = 'C:/Users/nadav/Final-Project/graph_data_set/deezer_edges.json'
f_target = 'C:/Users/nadav/Final-Project/graph_data_set/deezer_target.csv'

target = pd.read_csv(f_target)
graph_dics = read_json_file(file)

graph_features = defaultdict(list)
for i in range(0,len(graph_dics)):
    index = str(i)
    G = nx.Graph()
    G.add_edges_from(graph_dics[index])
    deg = [i[1] for i in G.degree]
    num_leafs = sum([i[1] if i[1] == 1 else 0 for i in G.degree])
    av_deg = sum(deg)/len(deg)
    graph_features['average_degree'].append(av_deg)
    graph_features['number_of_nodes'].append(G.number_of_nodes())
    graph_features['number_of_edges'].append(G.number_of_edges())
    graph_features['edges/nodes'].append(G.number_of_edges()/G.number_of_nodes())
    graph_features['num_leafs'].append(num_leafs)
    graph_features['density'].append(nx.density(G))

df = pd.DataFrame(graph_features)

path = 'C:/Users/nadav/Final-Project/graph_data_set/'
file_names = [fn[:-4] for fn in os.listdir(path) if fn.endswith(".txt")]

names = [fn[:-4] for fn in os.listdir(path) if fn.endswith(".txt")]
for name in file_names:
    values = [float(v.strip()) for v in open(path + f"{name}.txt").readlines()]
    df[name] = values


cliques3 = Cliques(3)
cliques4 = Cliques(4)
cliques3.fit_transform(nx.adj_matrix(G))
cliques4.fit_transform(nx.adj_matrix(G))
louv = Louvain()
size_center_cluster = []
num_clusters_louvain = []
cliques3list = []
cliques4list = []
for i in range(0,len(graph_dics)):
    index = str(i)
    G = nx.Graph()
    G.add_edges_from(graph_dics[index])
    adj = nx.adj_matrix(G)
    center_ind = nx.algorithms.barycenter(G)[0]
    cliques3list.append(cliques3.fit_transform(adj))
    cliques4list.append(cliques4.fit_transform(adj))
    G_clustering = louv.fit_transform(adj)
    center_cluster = G_clustering[center_ind] 
    uniq, counts = np.unique(G_clustering, return_counts=True)
    size_center_cluster.append(counts[center_cluster])
    num_clusters_louvain.append(len(uniq))
    
num_clusters_louvain = np.array(num_clusters_louvain)
size_center_cluster = np.array(size_center_cluster)
cliques3list = np.array(cliques3list)
cliques4list = np.array(cliques4list)

df['num_clusters_louvain'] = num_clusters_louvain
df['size_center_cluster'] = size_center_cluster
df['cliques3'] = cliques3list
df['cliques4'] = cliques4list



X = df
scaler = MinMaxScaler()
X = scaler.fit_transform(X) 
y = target['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Instantiate the clustering model and visualizer
model = KMeans()
my_title = " "
visualizer = KElbowVisualizer(model, k=(1,11),title =my_title, size =(400,300),timings=False)

visualizer.fit(X_train) 

visualizer.ax.set_xlabel('K',fontsize=12)
visualizer.ax.set_ylabel('distribution score',fontsize=14)
visualizer.ax.grid(False)
for tick in visualizer.ax.xaxis.get_majorticklabels():  
    tick.set_fontsize(12) 
for tick in visualizer.ax.yaxis.get_majorticklabels():  
    tick.set_fontsize(12)  
visualizer.ax.figure.savefig('elbow_kmeans_graphs.png', dpi=300)


def GMM(n_clusters, X):
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(X)

def kmeans(n_clusters, X):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)

def agglomerative(n_clusters, X):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    return agg.fit_predict(X)

def birch(n_clusters, X):
    bir = Birch(n_clusters=n_clusters)
    return bir.fit_predict(X)

def bayesian_gaussian(n_clusters, X):
    bgm = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
    return bgm.fit_predict(X)

n_clusters = visualizer.elbow_value_

MI_score = defaultdict(list)
Silhouette_Score_dict = defaultdict(list)
MI_X_SilhouetteScore = defaultdict(list)
for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    for cluster in [bayesian_gaussian, agglomerative, kmeans, birch]:
        X_ = X_train[smaple_index]
        y_ = y_train.values[smaple_index]
        MI = mutual_info_score(y_, cluster(n_clusters ,X_))
        MI_score[cluster.__name__].append(MI)
        ss = silhouette_score(X_,cluster(n_clusters ,X_))
        Silhouette_Score_dict[cluster.__name__].append(ss)
        MI_X_SilhouetteScore[cluster.__name__].append(2/((1/MI)+(1/ss)))

statisical_mat_ev = []
for i, sc in enumerate([MI_score ,Silhouette_Score_dict, MI_X_SilhouetteScore]):
    
    val1 = f_oneway(sc['kmeans'],sc['agglomerative'],
         sc['bayesian_gaussian'],sc['birch']).pvalue
    if val1 < 0.01:
        val1 = f"{val1:.2e}"
    else:
        val1 = round(val1, 2)
    val2 = ttest_rel(sc['birch'],sc['bayesian_gaussian']).pvalue
    if val2 < 0.01:
        val2 = f"{val2:.2e}"
    else:
        val2 = round(val2, 2)
    a = [val1, val2]
    statisical_mat_ev.append(a)

# Numbers of pairs of bars you want
N = 4

# Data on X-axis

# Specify the values of blue bars (height)
blue_bar = (np.mean(MI_score['kmeans']),np.mean(MI_score['bayesian_gaussian'])
            ,np.mean(MI_score['birch']),np.mean(MI_score['agglomerative']))
# Specify the values of orange bars (height)
orange_bar = (np.mean(MI_X_SilhouetteScore['kmeans']), np.mean(MI_X_SilhouetteScore['bayesian_gaussian']), 
             np.mean(MI_X_SilhouetteScore['birch']),np.mean(MI_X_SilhouetteScore['agglomerative']))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='MI')
plt.bar(ind + width, orange_bar, width, label='MI & SilhouetteScore')

plt.ylabel('Average MI',size =14)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('KMeans', 'Bayesian Gaussian', 'Birch', 'Agglomerative'), size = 14)
plt.yticks(size = 14)
# Finding the best position for legends and putting it
plt.legend(loc='best',fontsize = 12 )
plt.grid(False)
plt.savefig('graph_bars.png', dpi=300)

X_ = X_test
y_ = y_test.values
MI_score_test = mutual_info_score(y_, bayesian_gaussian(n_clusters ,X_))
Silhouette_Score_dict_test = silhouette_score(X_,bayesian_gaussian(n_clusters ,X_))
MI_X_SilhouetteScore_test = ss*MI

#print(f" MI-test:{MI_score_test},  Silhouette test:{Silhouette_Score_dict_test},  MI X Silhouette test:{MI_X_SilhouetteScore_test}")



def GMM_AD(X, anomaly_perc):
    gmm = GaussianMixture(n_components = 6 ,random_state=42)
    gmm.fit(X)
    score = gmm.score_samples(X)
    threshold = np.percentile(score, 100*anomaly_perc)
    anomaly = score < threshold
    if anomaly.sum() == 0:
        return None
    return anomaly

def LOF_AD(X, anomaly_perc):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=anomaly_perc)
    y_pred = lof.fit_predict(X)
    anomaly = (y_pred == -1)
    if anomaly.sum() == 0:
        return None
    return anomaly

def OneCSVM_AD(X, anomaly_perc):
    ocsvm = OneClassSVM(gamma='auto', nu=anomaly_perc)
    y_pred = ocsvm.fit_predict(X)
    anomaly = y_pred == -1
    if anomaly.sum() == 0:
        return None
    return anomaly
    

n_clusters = visualizer.elbow_value_

gmm = GaussianMixture(n_components = n_clusters ,random_state=42)
gmm.fit(X_train)
score = gmm.score_samples(X_train)
threshold = score.mean() - 3*score.std()
percentile = np.mean(score < threshold)

results = defaultdict(list)
anomalys_dict = defaultdict(list)
anomalys_ev = defaultdict(lambda: defaultdict(list))

for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.5).split(X_train):
    X = X_train[smaple_index]
    y = y_train.values[smaple_index]

    for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
        anomaly = func(X, percentile)
        if anomaly is None:
            continue
        anomalys_ev['gender'][(func.__name__+ "_MI")].append(mutual_info_score(anomaly,y))

        MI = adjusted_mutual_info_score(y[~anomaly] ,kmeans(n_clusters, X[~anomaly]))
        results[(func.__name__+ "_MI")].append(MI)

X = X_train
y = y_train.values

for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
    anomaly = func(X, percentile)
    if anomaly is None:
        continue
    anomalys_dict[func.__name__] = anomaly



MI_Dimension_Reduction = defaultdict(lambda: defaultdict(list))

for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    X = X_train[smaple_index]
    y = y_train.values[smaple_index]
    MI = mutual_info_score(y ,bayesian_gaussian(n_clusters, X))
    MI_Dimension_Reduction['regular data'][0].append(MI)
    for k in [2,4,6,8,12,16,20]:
        fa = FeatureAgglomeration(n_clusters=k)
        pca = PCA(n_components=k)
        X_fa = fa.fit_transform(X)
        X_pca = pca.fit_transform(X)
        MI_reduced_fa = mutual_info_score(y ,bayesian_gaussian(n_clusters, X_fa))
        MI_reduced_pca = mutual_info_score(y ,bayesian_gaussian(n_clusters, X_pca))
        MI_Dimension_Reduction['FeatureAgglomeration'][k].append(MI_reduced_fa)
        MI_Dimension_Reduction['PCA'][k].append(MI_reduced_pca)

 
best_DR = 'regular data'
best_D = 0
best_score = 0
for key in MI_Dimension_Reduction:
        for k in MI_Dimension_Reduction[key]:
            cur_score = sum(MI_Dimension_Reduction[key][k])
            if cur_score > best_score:
                best_score = cur_score
                best_D = k
                best_DR = key
#print("best dimention reduction:"+best_DR+", with "+ str(best_D)+"dimension")


X_ = X_test
y_ = y_test.values
pca = PCA(n_components=best_D)
X_pca = pca.fit_transform(X_)
MI_score_test = mutual_info_score(y_, bayesian_gaussian(n_clusters ,X_pca))
#print(f" MI-test with dimension reduction:{MI_score_test}")


reducer = umap.UMAP()
label = y_train
X = X_train
pca = PCA(n_components=best_D)
X = pca.fit_transform(X)

cluster_label = bayesian_gaussian(n_clusters, X)
embedding = reducer.fit_transform(X, cluster_label)

fig, ax = plt.subplots()

scat1 = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_label, cmap="Spectral", s=180, alpha=0.7)
scat2 = ax.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap="Set3", s=10)

legend1 = plt.legend(*scat1.legend_elements(), loc="lower right", title="Cluster", frameon=True)
ax.add_artist(legend1)

legend2 = plt.legend(*scat2.legend_elements(), loc="best", title="Gender", frameon=True)
ax.add_artist(legend2)

for cluster in range(n_clusters):
    x = np.mean(embedding[cluster_label==cluster, 0])
    y = np.mean(embedding[cluster_label==cluster, 1])
    plt.scatter(x, y, 
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k")
    plt.scatter(x, y, marker="$%d$" % cluster, alpha=1, s=50, edgecolor="k")
    
plt.savefig('umap_graph.png', dpi=300)