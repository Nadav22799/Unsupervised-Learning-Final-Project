import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from collections import defaultdict
from sklearn.cluster import Birch, AgglomerativeClustering, KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import f_oneway, ttest_rel
from sklearn.model_selection import ShuffleSplit

from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

import umap
import pandas as pd

import utils

feature_columns = 'mfcc'
my_title = " "


plt.rcParams['figure.figsize'] = (17, 5)

# Load metadata and features.
tracks = utils.load('fma_metadata/tracks.csv')
features = utils.load('fma_metadata/features.csv')

np.testing.assert_array_equal(features.index, tracks.index)


data = tracks['set', 'subset'] <= 'large'

train = (tracks['set', 'split'] == 'training') & tracks['track', 'genre_top'].notna()
val = (tracks['set', 'split'] == 'validation') & tracks['track', 'genre_top'].notna()
test = (tracks['set', 'split'] == 'test') & tracks['track', 'genre_top'].notna()

y_train = tracks.loc[data & train, ('track', 'genre_top')]
y_test = tracks.loc[data & test, ('track', 'genre_top')]
ex_var_train = tracks.loc[data & train, ('artist', 'name')]
ex_var_test = tracks.loc[data & test, ('artist', 'name')] 

labeler_y = skl.preprocessing.LabelEncoder().fit(y_train)
y_train = labeler_y.transform(y_train)
y_test = labeler_y.transform(y_test)

labeler_ex = skl.preprocessing.LabelEncoder().fit(ex_var_test.to_list() + ex_var_train.to_list())
ex_var_train = labeler_ex.transform(ex_var_train)
ex_var_test = labeler_ex.transform(ex_var_test)

X_train = features.loc[data & train]
X_test = features.loc[data & test]

columns = [['mfcc'], ['chroma_cens'], ['tonnetz'], ['spectral_contrast']]
columns.append(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'])
columns.append(['rmse', 'zcr'])
score_cols = defaultdict(list)
km = KMeans(random_state=42)
for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    for col in columns:
        X = pd.concat((X_train[feature_name] for feature_name in col), axis=1)
        MI = adjusted_mutual_info_score(y_train[smaple_index], km.fit_predict(X.values[smaple_index]))
        score_cols[col[0]].append(MI)
     

feature_columns = 'mfcc'
my_title = " "

# Instantiate the clustering model and visualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,16),title =my_title, size =(400,300),timings=False)

visualizer.fit(X_train[feature_columns])       
 

visualizer.ax.set_xlabel('K',fontsize=12)
visualizer.ax.set_ylabel('distribution score',fontsize=14)
visualizer.ax.grid(False)
for tick in visualizer.ax.xaxis.get_majorticklabels():  
    tick.set_fontsize(12) 
for tick in visualizer.ax.yaxis.get_majorticklabels():  
    tick.set_fontsize(12)  
visualizer.ax.figure.savefig('elbow_kmeans_music.png', dpi=300)


def kmeans(n_clusters, X):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)
def GMM(n_clusters, X):
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(X)
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

MI_scores = defaultdict(lambda: defaultdict(list))
for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    for i, ev in enumerate(['genre_top','name']):
        for cluster in [kmeans,bayesian_gaussian, agglomerative, birch]:
            X = X_train[feature_columns].iloc[smaple_index]
            y = [y_train[smaple_index], ex_var_train[smaple_index]]
            MI = adjusted_mutual_info_score(y[i], cluster(n_clusters ,X))
            MI_scores[ev][cluster.__name__].append(MI)

# Numbers of pairs of bars you want
N = 4

# Data on X-axis

# Specify the values of blue bars (height)
blue_bar = (np.mean(MI_scores['genre_top']['kmeans']),np.mean(MI_scores['genre_top']['bayesian_gaussian'])
            ,np.mean(MI_scores['genre_top']['birch']),np.mean(MI_scores['genre_top']['agglomerative']))
# Specify the values of orange bars (height)
orange_bar = (np.mean(MI_scores['name']['kmeans']), np.mean(MI_scores['name']['bayesian_gaussian']), 
             np.mean(MI_scores['name']['birch']),np.mean(MI_scores['name']['agglomerative']))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Genre Top')
plt.bar(ind + width, orange_bar, width, label='Artist Name')

plt.ylabel('Average MI',size =14)

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('KMeans', 'Bayesian Gaussian', 'Birch', 'Agglomerative'), size = 14)
plt.yticks(size = 14)
# Finding the best position for legends and putting it
plt.legend(loc='best',fontsize = 12 )
plt.grid(False)
plt.savefig('music_bars.png', dpi=300)

statisical_mat_ev = []
for i, ev in enumerate(MI_scores):
    val1 = f_oneway(MI_scores[ev]['kmeans'],MI_scores[ev]['agglomerative'],
         MI_scores[ev]['bayesian_gaussian'],MI_scores[ev]['birch']).pvalue
    if val1 < 0.01:
        val1 = f"{val1:.2e}"
    else:
        val1 = round(val1, 2)
    val2 = ttest_rel(MI_scores[ev]['kmeans'],MI_scores[ev]['bayesian_gaussian']).pvalue
    if val2 < 0.01:
        val2 = f"{val2:.2e}"
    else:
        val2 = round(val2, 2)
    a = [val1, val2]
    statisical_mat_ev.append(a)

statisical_mat_cl = []
for i, cluster in enumerate([kmeans , bayesian_gaussian, birch, agglomerative]):
    val = ttest_rel(MI_scores['genre_top'][cluster.__name__],MI_scores['name'][cluster.__name__]).pvalue
    if val < 0.01:
        val = f"{val:.2e}"
    else:
        val = round(val, 2)
    statisical_mat_cl.append([val])

# print(statisical_mat_ev)
# print(statisical_mat_ev)



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
gmm.fit(X_train[feature_columns])
score = gmm.score_samples(X_train[feature_columns])
threshold = score.mean() - 3*score.std()
percentile = np.mean(score < threshold)

results = defaultdict(list)
anomalys_dict = defaultdict(list)
anomalys_ev = defaultdict(lambda: defaultdict(list))

for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    X = X_train[feature_columns].iloc[smaple_index]
    y = y_train[smaple_index]
    ev = ex_var_train[smaple_index]

    for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
        anomaly = func(X, percentile)
        if anomaly is None:
            continue
        anomalys_ev['genere'][(func.__name__+ "_MI")].append(adjusted_mutual_info_score(anomaly, y))
        anomalys_ev['name'][(func.__name__+ "_MI")].append(adjusted_mutual_info_score(anomaly, ev))

        MI = adjusted_mutual_info_score(y[~anomaly] ,kmeans(n_clusters, X[~anomaly]))
        results[(func.__name__+ "_MI")].append(MI)
    
X = X_train[feature_columns]
y = y_train
ev = ex_var_train

for func in [GMM_AD, LOF_AD, OneCSVM_AD]:
    anomaly = func(X, percentile)
    if anomaly is None:
        continue
    anomalys_dict[func.__name__] = anomaly



MI_Dimension_Reduction = defaultdict(lambda: defaultdict(list))

n_clusters = visualizer.elbow_value_
for _, smaple_index in ShuffleSplit(n_splits=30, random_state=42, test_size=0.1).split(X_train):
    X = X_train[feature_columns].iloc[smaple_index]
    y = y_train[smaple_index]   
    MI = adjusted_mutual_info_score(y ,kmeans(n_clusters, X))
    MI_Dimension_Reduction['regular data'][0].append(MI)
    for k in [2,4,8,16,32,64,128]:
        pca = PCA(n_components=k)
        fa = FeatureAgglomeration(n_clusters=k)
        X_pca = pca.fit_transform(X)
        X_fa = fa.fit_transform(X)
        MI_reduced_fa = adjusted_mutual_info_score(y ,kmeans(n_clusters, X_fa))
        MI_reduced_pca = adjusted_mutual_info_score(y ,kmeans(n_clusters, X_pca))
        MI_Dimension_Reduction['FeatureAgglomeration'][k].append(MI_reduced_fa)
        MI_Dimension_Reduction['PCA'][k].append(MI_reduced_pca)

for key in ['FeatureAgglomeration', 'PCA']:
    best_score = 0
    best_score_key = 0

    for k in MI_Dimension_Reduction[key]:
        cur_score = sum(MI_Dimension_Reduction[key][k])
        if cur_score > best_score:
            best_score = cur_score
            best_score_key = k
#    print(key, best_score, best_score_key)


reducer = umap.UMAP()
label = y_test
X = X_test

cluster_label = kmeans(n_clusters, X)
embedding = reducer.fit_transform(X)

fig, ax = plt.subplots()

scat1 = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_label, cmap="Set3", s=180, alpha=0.5)
scat2 = ax.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap="Spectral", s=10)

legend1 = plt.legend(*scat1.legend_elements(), loc="lower right", title="Cluster", frameon=True)
ax.add_artist(legend1)


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
    
plt.savefig('umap_music.png', dpi=300)


