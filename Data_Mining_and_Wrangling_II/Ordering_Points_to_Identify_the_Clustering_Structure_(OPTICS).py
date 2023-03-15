rng = np.random.default_rng(1337)
X1 = rng.random(500)*6*np.pi
Y1 = np.cos(X1) + rng.random(500) + 1
X2 = rng.random(500)*6*np.pi
Y2 = np.cos(X2) + rng.random(500) - 1
X3 = rng.random(200)*6*np.pi
Y3 = rng.random(200)*5 - 2
X = np.concatenate((X1, X2, X3))
Y = np.concatenate((Y1, Y2, Y3))
XY = np.vstack((X, Y)).T
plt.scatter(X, Y, s=0.6);


from sklearn.cluster import OPTICS, cluster_optics_dbscan
optics = OPTICS(min_samples=50)
optics.fit(XY)
plt.plot(optics.reachability_[optics.ordering_])
plt.ylabel('reachability');


cluster_labels = cluster_optics_dbscan(
    reachability=optics.reachability_,
    core_distances=optics.core_distances_,
    ordering=optics.ordering_,
    eps=0.8
)
plt.scatter(X, Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
print('Silhouette score:', silhouette_score(XY, cluster_labels))


cluster_labels = cluster_optics_dbscan(
    reachability=optics.reachability_,
    core_distances=optics.core_distances_,
    ordering=optics.ordering_,
    eps=1
)
plt.scatter(X, Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
print('Silhouette score:', silhouette_score(XY, cluster_labels))


cluster_labels = optics.labels_
plt.scatter(X, Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
print('Silhouette score:', silhouette_score(XY, cluster_labels))


data_wine = load_wine()
standard_scaler = StandardScaler()
X_wine = standard_scaler.fit_transform(data_wine['data'])
target_wine = data_wine['target']
wine_pca = PCA(n_components=2, random_state=1337)
X_wine_new = wine_pca.fit_transform(X_wine)


from sklearn.cluster import OPTICS, cluster_optics_dbscan

wine_X, wine_Y = X_wine_new[:, 0], X_wine_new[:, 1]

# Since the wines dataset has fewer data points compared to our
# previous example we have to set the min_samples lower at 8
optics = OPTICS(min_samples=8) 
optics.fit(X_wine_new)
plt.plot(optics.reachability_[optics.ordering_])
plt.ylabel('reachability');


epss = np.arange(0.4, 1.1, 0.1)

for eps in epss:
    cluster_labels = cluster_optics_dbscan(
        reachability=optics.reachability_,
        core_distances=optics.core_distances_,
        ordering=optics.ordering_,
        eps=eps
    )
    plt.scatter(wine_X, wine_Y, c=cluster_labels, s=0.6)
    print(f'\nOPTICS epsilon = {eps}')
    plt.show()
    print('Number of clusters:', cluster_labels.max()+1)
    print('Number of noise points:', (cluster_labels==-1).sum())
    print('Number of points:', len(cluster_labels))
    print('Silhouette score:', silhouette_score(X_wine_new, cluster_labels))
    
    
cluster_labels = optics.labels_
plt.scatter(wine_X, wine_Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
print('Silhouette score:', silhouette_score(X_wine_new, cluster_labels))


data_newsgroups = fetch_20newsgroups(
    subset='train', 
    categories=['comp.graphics', 'rec.autos'],
    shuffle=False, 
    remove=['headers', 'footers', 'quotes'])
tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-z-]+', 
                                   stop_words='english',
                                   min_df=5)
bow_ng = tfidf_vectorizer.fit_transform(data_newsgroups['data'])
nonzeros = bow_ng.sum(axis=1).nonzero()[0]
bow_ng = bow_ng[nonzeros]
target_ng = data_newsgroups['target'][nonzeros]
ng_svd = TruncatedSVD(n_components=2, random_state=1337)
X_ng_new = ng_svd.fit_transform(bow_ng)


from sklearn.cluster import OPTICS, cluster_optics_dbscan

ng_X, ng_Y = X_ng_new[:, 0], X_ng_new[:, 1]

# Since the newsgroups dataset has almost the same number
# of data points compared to our previous example we have to 
# set the min_samples around the same value which is 50
optics = OPTICS(min_samples=50) 
optics.fit(X_ng_new)
plt.plot(optics.reachability_[optics.ordering_])
plt.ylabel('reachability');


epss = np.arange(0.02, 0.06, 0.01)

for eps in epss:
    cluster_labels = cluster_optics_dbscan(
        reachability=optics.reachability_,
        core_distances=optics.core_distances_,
        ordering=optics.ordering_,
        eps=eps
    )
    plt.scatter(ng_X, ng_Y, c=cluster_labels, s=0.6)
    print(f'\nOPTICS epsilon = {eps}')
    plt.show()
    print('Number of clusters:', cluster_labels.max()+1)
    print('Number of noise points:', (cluster_labels==-1).sum())
    print('Number of points:', len(cluster_labels))
#     print('Silhouette score:', silhouette_score(X_ng_new, cluster_labels))


cluster_labels = optics.labels_
plt.scatter(ng_X, ng_Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
# print('Silhouette score:', silhouette_score(X_ng_new, cluster_labels))