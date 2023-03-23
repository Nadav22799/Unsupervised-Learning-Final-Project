import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy.stats import f_oneway, ttest_rel

from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration, Birch
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from yellowbrick.cluster import KElbowVisualizer

import umap
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"

All_concentration = {}
All_labels = {} 
All_features = {}
for i in range(1,11):
    batch_end = str(i)+'.dat'
    with open('C:/Users/nadav/Final-Project/gas_data_set/driftdataset/batch'+ batch_end) as f:
        label = [] 
        concentration = []
        features = []
        for line in f:
            ls = line.split(" ")
            gas = int(ls[0][0])
            label.append(gas)
            concentration.append(float(ls[0].split(";")[1]))
            cur_feat = [float(ls[i].split(":")[1]) for i in range(1,129)]
            features.append(cur_feat)

    All_features[i] = np.array(features)  
    All_labels[i] = np.array(label) 
    All_concentration[i] = np.array(concentration) 


X_train = np.concatenate([All_features[batch] for batch in range(1,7)], axis=0)
X_train = MinMaxScaler().fit_transform(X_train.T).T

model = KMeans()
my_title = " "
visualizer = KElbowVisualizer(model, k=(2,16),title =my_title, size =(400,300),timings=False)

visualizer.ax.set_xlabel('K',fontsize=12)
visualizer.ax.set_ylabel('distribution score',fontsize=14)
visualizer.ax.grid(False)
for tick in visualizer.ax.xaxis.get_majorticklabels():  
    tick.set_fontsize(12) 
for tick in visualizer.ax.yaxis.get_majorticklabels():  
    tick.set_fontsize(12)  
visualizer.fit(X_train)
visualizer.ax.figure.savefig('elbow_kmeans_gas.png', dpi=300)

        

def kmeans(n_clusters, X):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)
def birch(n_clusters, X):
    bir = Birch(n_clusters=n_clusters)
    return bir.fit_predict(X)
def agglomerative(n_clusters, X):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    return agg.fit_predict(X)
def bayesian_gaussian(n_clusters, X):
    bgm = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
    return bgm.fit_predict(X)
def GMM(n_clusters, X):
    gm = GaussianMixture(n_components=n_clusters)
    return gm.fit_predict(X)


n_clusters = visualizer.elbow_value_
MI_scaling = defaultdict(list)
for batch in range(1,7):
    for _ in range(5):
        X_Sample, _, y_type, _, y_concen, _ = train_test_split(
            All_features[batch], All_labels[batch], All_concentration[batch], test_size=0.5, shuffle=True, random_state=42
        )
        X_Sample_scaled = MinMaxScaler().fit_transform(X_Sample.T).T
        X_Sample_regular = X_Sample
        MI_regular = mutual_info_score(y_type, kmeans(n_clusters, X_Sample_regular))
        MI_scaling['regular'].append(MI_regular) 
        MI_scaled = mutual_info_score(y_type, kmeans(n_clusters, X_Sample_scaled))
        MI_scaling['minmaxscale'].append(MI_scaled) 

n_clusters = visualizer.elbow_value_
MI_scores = defaultdict(lambda: defaultdict(list))
for batch in range(1,7):
    for cluster in [kmeans , bayesian_gaussian, birch, agglomerative]:
        for _ in range(5):
            X_Sample, _, y_type, _, y_concen, _ = train_test_split(
                All_features[batch], All_labels[batch], All_concentration[batch], test_size=0.5, shuffle=True, random_state=42
            )
            X_Sample = MinMaxScaler().fit_transform(X_Sample.T).T
            MI_type = mutual_info_score(y_type, cluster(n_clusters, X_Sample))
            MI_scores['type'][cluster.__name__].append(MI_type) 
            MI_concentration = mutual_info_score(y_concen, cluster(n_clusters, X_Sample))
            MI_scores['concentration'][cluster.__name__].append(MI_concentration) 

# Numbers of pairs of bars you want
N = 4

# Data on X-axis

# Specify the values of blue bars (height)
blue_bar = (np.mean(MI_scores['type']['kmeans']),np.mean(MI_scores['type']['bayesian_gaussian'])
            ,np.mean(MI_scores['type']['birch']),np.mean(MI_scores['type']['agglomerative']))
# Specify the values of orange bars (height)
orange_bar = (np.mean(MI_scores['concentration']['kmeans']), np.mean(MI_scores['concentration']['bayesian_gaussian']), 
             np.mean(MI_scores['concentration']['birch']),np.mean(MI_scores['concentration']['agglomerative']))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Gas Type')
plt.bar(ind + width, orange_bar, width, label='Gas Concentration')

plt.ylabel('Average MI',size =14)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('KMeans', 'Bayesian Gaussian', 'Birch', 'Agglomerative'), size = 14)
plt.yticks(size = 14)

# Finding the best position for legends and putting it
plt.legend(loc='best',fontsize = 12 )
plt.grid(False)
plt.savefig('gas_bars.png', dpi=300)

MI_best = []
for batch in range(7,11):
    X_test = All_features[batch]
    X_test = MinMaxScaler().fit_transform(X_test.T).T
    MI = mutual_info_score(All_labels[batch], bayesian_gaussian(n_clusters, X_test))
    MI_best.append(MI)     

statisical_mat_ev = []
for i, ev in enumerate(MI_scores):
    val1 = f_oneway(MI_scores[ev]['kmeans'],MI_scores[ev]['agglomerative'],
         MI_scores[ev]['bayesian_gaussian'],MI_scores[ev]['birch']).pvalue
    if val1 < 0.01:
        val1 = f"{val1:.2e}"
    else:
        val1 = round(val1, 3)
    val2 = ttest_rel(MI_scores[ev]['kmeans'],MI_scores[ev]['bayesian_gaussian']).pvalue
    if val2 < 0.01:
        val2 = f"{val2:.2e}"
    else:
        val2 = round(val2, 2)
    a = [val1, val2]
    statisical_mat_ev.append(a)

statisical_mat_cl = []
for i, cluster in enumerate([kmeans , bayesian_gaussian, birch, agglomerative]):
    val = ttest_rel(MI_scores['type'][cluster.__name__],MI_scores['concentration'][cluster.__name__]).pvalue
    if val < 0.01:
        val = f"{val:.2e}"
    else:
        val = round(val, 2)
    statisical_mat_cl.append([val])

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

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
    
percentiles = []

for batch in range(1,7):
    gmm = GaussianMixture(n_components = n_clusters ,random_state=42)
    data  = MinMaxScaler().fit_transform(All_features[batch].T).T
    gmm.fit(data)
    score = gmm.score_samples(data)
    threshold = score.mean() - 3*score.std()
    perc = np.mean(score < threshold)
    percentiles.append(perc)
    
percentile = np.mean(percentiles)

anomalys_ev = defaultdict(lambda: defaultdict(list))
results = defaultdict(list)

for batch in range(1,7):
    for _ in range(5):
        X_Sample, _, y_type, _, y_concen, _ = train_test_split(
                All_features[batch], All_labels[batch], All_concentration[batch], test_size=0.5, shuffle=True, random_state=42
            )
        X_Sample = MinMaxScaler().fit_transform(X_Sample.T).T
        
        for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
            anomaly = func(X_Sample, percentile)
            if anomaly is None:
                continue
            
            anomalys_ev['gas_type'][(func.__name__+ "_MI")].append(mutual_info_score(anomaly, y_type))
            anomalys_ev['gas_concentration'][(func.__name__+ "_MI")].append(mutual_info_score(anomaly, y_concen))

            MI = mutual_info_score(y_type[~anomaly] ,bayesian_gaussian(n_clusters, X_Sample[~anomaly]))
            results[(func.__name__+ "_MI")].append(MI)



anomalys_dict = defaultdict(lambda: defaultdict(list))

for batch in range(1,7):
    for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
        gmm = GaussianMixture(n_components = n_clusters ,random_state=42)
        X = MinMaxScaler().fit_transform(All_features[batch].T).T
        anomaly = func(X, percentile)
        if anomaly is None:
            continue
        anomalys_dict[func.__name__][batch] = anomaly

MI_Dimension_Reduction = defaultdict(lambda: defaultdict(list))
ev_str = ['gas','concentration']
for batch in range(1,7):
    for _ in range(5):
        X_Sample, _, y_type, _, y_concen, _ = train_test_split(
            All_features[batch], All_labels[batch], All_concentration[batch], test_size=0.5, shuffle=True, random_state=42
        )
        X_Sample = MinMaxScaler().fit_transform(X_Sample.T).T
        y_train = [y_type, y_concen]
        for i, ev in enumerate(y_train):
            MI = mutual_info_score(ev ,agglomerative(n_clusters, X_Sample))
            MI_Dimension_Reduction['regular data_'+ ev_str[i]][0].append(MI)
            for k in [2,4,8,16,32]:
                fa = FeatureAgglomeration(n_clusters=k)
                pca = PCA(n_components=k)
                X_fa = fa.fit_transform(X_Sample)
                X_pca = pca.fit_transform(X_Sample)
                MI_reduced_fa = mutual_info_score(ev ,bayesian_gaussian(n_clusters, X_fa))
                MI_reduced_pca = mutual_info_score(ev ,bayesian_gaussian(n_clusters, X_pca))
                MI_Dimension_Reduction['FeatureAgglomeration_'+ ev_str[i]][k].append(MI_reduced_fa)
                MI_Dimension_Reduction['PCA_'+ ev_str[i]][k].append(MI_reduced_pca)
    
best_DR_gas = 'regular data_gas'
best_D = 0
best_score = 0
for key in MI_Dimension_Reduction:
    if key[-1] =='s':
        for k in MI_Dimension_Reduction[key]:
            cur_score = sum(MI_Dimension_Reduction[key][k])
            if cur_score > best_score:
                best_score = cur_score
                best_D = k
                best_DR_gas = key
best_DR_gas, best_D
#print("best dimension reduction:"+best_DR_gas+"with"+str(best_D)+"dimensions")

MI_best_D = {}
for batch in range(7,11):
    X_test = All_features[batch]
    X_test = MinMaxScaler().fit_transform(X_test.T).T
    fa = FeatureAgglomeration(n_clusters=16)
    X_fa = fa.fit_transform(X_test)
    MI = mutual_info_score(All_labels[batch], bayesian_gaussian(n_clusters, X_fa))
    MI_best_D[batch] = MI  
#print("result on test with dimension reduction:"+ str(MI_best_D))


batch = 6
reducer = umap.UMAP(random_state=42)
label = All_labels[batch]
X = All_features[batch]
X = MinMaxScaler().fit_transform(X.T).T
fa = FeatureAgglomeration(n_clusters=best_D)
X = fa.fit_transform(X)
n_clusters = 5

cluster_label = bayesian_gaussian(n_clusters, X)
embedding = reducer.fit_transform(X, cluster_label)

fig, ax = plt.subplots()

scat1 = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_label, cmap="Set3", s=180, alpha=0.5)
scat2 = ax.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap="Spectral", s=10)

legend1 = plt.legend(*scat1.legend_elements(), loc="lower right", title="Cluster", bbox_to_anchor=(1.11, 0), frameon=True)
ax.add_artist(legend1)

legend2 = plt.legend(*scat2.legend_elements(), loc="upper right", title="Gas", bbox_to_anchor=(1.11, 1), frameon=True)
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
    

plt.savefig('umap_gas.png', dpi=300)

reducer = umap.UMAP(random_state=42)
label = All_labels[batch]
X = All_features[batch]
X = MinMaxScaler().fit_transform(X.T).T
fa = FeatureAgglomeration(n_clusters=best_D)
X = fa.fit_transform(X)
n_clusters = 5

embedding = reducer.fit_transform(X, cluster_label)

fig, ax = plt.subplots()

scat2 = ax.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap="Spectral", s=20) 
    
legend2 = plt.legend(*scat2.legend_elements(), loc="upper right", title="Gas", bbox_to_anchor=(1.11, 1), frameon=True)
ax.add_artist(legend2)

anomalies = anomalys_dict['GMM_AD'][batch]
x = embedding[anomalies, 0]
y = embedding[anomalies, 1]
scat1 = ax.scatter(x, y, 
    marker="o",
    c=[1 for _ in x],
    cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),
    alpha=0.4,
    s=100,
    edgecolor="k",
    linewidths=2)
h, l = scat1.legend_elements()
legend1 = plt.legend(h, ['Anomalies'], loc="best",  frameon=True)
ax.add_artist(legend1)
    
plt.savefig('umap_gas_outliers.png', dpi=300)


