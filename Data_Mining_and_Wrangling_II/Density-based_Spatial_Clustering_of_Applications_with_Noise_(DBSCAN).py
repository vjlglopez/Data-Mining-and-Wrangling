rng = np.random.default_rng(1337)
X1 = rng.random(500)*6*np.pi
Y1 = np.cos(X1) + rng.random(500) + 1
X2 = rng.random(500)*6*np.pi
Y2 = np.cos(X2) + rng.random(500) - 1
X3 = rng.random(200)*6*np.pi
Y3 = rng.random(200)*5 - 2
X = np.concatenate((X1, X2, X3))
Y = np.concatenate((Y1, Y2, Y3))
plt.scatter(X, Y, s=0.6);


XY = np.vstack((X, Y)).T
plt.scatter(X, Y, c=KMeans(2).fit_predict(XY), s=0.6);


from sklearn.neighbors import NearestNeighbors

def ave_nn_dist(n_neighbors, data):
    nearest = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances = nearest.kneighbors(return_distance=True)
    return np.sort(np.mean(distances[0], axis=1)).tolist()


plt.plot(ave_nn_dist(4, XY))
plt.xlabel('point')
plt.ylabel('distance');


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=4)
cluster_labels = dbscan.fit_predict(XY)
plt.scatter(X, Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))


from DBCV import DBCV
print('Silhouette score:', silhouette_score(XY, cluster_labels))
print('DBCV:', DBCV(XY, cluster_labels, dist_function=euclidean))


epss = np.arange(0.1, 1.1, 0.1)
min_pts = 4
dbcvs = []
for eps in epss:
    cluster_labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(XY)
    dbcvs.append(DBCV(XY, cluster_labels, dist_function=euclidean))
plt.plot(epss, dbcvs, '-o')
plt.xlabel(r'$\epsilon$')
plt.ylabel('DBCV');


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=4)
cluster_labels = dbscan.fit_predict(XY)
plt.scatter(X, Y, c=cluster_labels, s=0.6)
print('Number of clusters:', cluster_labels.max()+1)
print('Number of noise points:', (cluster_labels==-1).sum())
print('Number of points:', len(cluster_labels))
print('Silhouette score:', silhouette_score(XY, cluster_labels))
print('DBCV:', DBCV(XY, cluster_labels, dist_function=euclidean))


data_wine = load_wine()
standard_scaler = StandardScaler()
X_wine = standard_scaler.fit_transform(data_wine['data'])
target_wine = data_wine['target']
wine_pca = PCA(n_components=2, random_state=1337)
X_wine_new = wine_pca.fit_transform(X_wine)


wine_X, wine_Y = X_wine_new[:, 0], X_wine_new[:, 1]
plt.scatter(wine_X, wine_Y, c=KMeans(3).fit_predict(X_wine_new), s=0.6)
plt.title('$k$-means with $k=3$.')
plt.show()

plt.plot(ave_nn_dist(3, X_wine_new))
plt.title(f'Average nearest neighbor distance = 3')
plt.xlabel('point')
plt.ylabel('distance')
plt.show()


epss = np.arange(0.4, 0.8, 0.1) # range where the elbow is prominent
samps = np.arange(3, 6, 1) # keeping the low number of samples

for eps in epss:
    for samp in samps:
        print(f'\nEpsilon = {eps}, Minimum Samples = {samp}')
        dbscan = DBSCAN(eps=eps, min_samples=samp)
        cluster_labels = dbscan.fit_predict(X_wine_new)
        plt.scatter(wine_X, wine_Y, c=cluster_labels, s=0.6)
        plt.show()
        print('Number of clusters:', cluster_labels.max()+1)
        print('Number of noise points:', (cluster_labels==-1).sum())
        print('Number of points:', len(cluster_labels))
        
        
epss = np.arange(0.4, 0.8, 0.1) # range where the elbow is prominent
dbcvs = []
for eps in epss:
    cluster_labels = DBSCAN(eps=eps, min_samples=4).fit_predict(X_wine_new)
    dbcvs.append(DBCV(X_wine_new, cluster_labels, dist_function=euclidean))

plt.plot(epss, dbcvs, '-o')
plt.xlabel(r'$\epsilon$')
plt.ylabel('DBCV');


# according to eyeballing the optimal hyperparameters are  0.5, Minimum Samples = 4
# according to DBCV results the optimal hyper parameters are  0.4, Minimum Samples = 4

opt_params = [0.4, 0.5]

for i in opt_params:
    dbscan = DBSCAN(eps=i, min_samples=4)
    cluster_labels = dbscan.fit_predict(X_wine_new)
    plt.scatter(wine_X, wine_Y, c=cluster_labels, s=0.6)
    plt.title(f'Epsilon = {i}')
    plt.show()
    print('Number of clusters:', cluster_labels.max()+1)
    print('Number of noise points:', (cluster_labels==-1).sum())
    print('Number of points:', len(cluster_labels))
    print('DBCV:', DBCV(X_wine_new, cluster_labels, dist_function=euclidean))
    
    
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


ng_X, ng_Y = X_ng_new[:, 0], X_ng_new[:, 1]
plt.scatter(ng_X, ng_Y, c=KMeans(2).fit_predict(X_ng_new), s=0.6)
plt.title('$k$-means with $k=2$.')
plt.show()

plt.plot(ave_nn_dist(4, X_ng_new))
plt.title(f'Average nearest neighbor distance = 3')
plt.xlabel('point')
plt.ylabel('distance')
plt.show()


epss = np.arange(0.01, 0.04, 0.01) # range where the elbow is prominent
samps = np.arange(3, 11, 1) # keeping the low number of samples

for eps in epss:
    for samp in samps:
        print(f'\nEpsilon = {eps}, Minimum Samples = {samp}')
        dbscan = DBSCAN(eps=eps, min_samples=samp)
        cluster_labels = dbscan.fit_predict(X_ng_new)
        plt.scatter(ng_X, ng_Y, c=cluster_labels, s=0.6)
        plt.show()
        print('Number of clusters:', cluster_labels.max()+1)
        print('Number of noise points:', (cluster_labels==-1).sum())
        print('Number of points:', len(cluster_labels))
        
        
epss = np.arange(0.01, 0.04, 0.01) # range where the elbow is prominent

dbcvs = []
for eps in epss:
    cluster_labels = DBSCAN(eps=eps, min_samples=7).fit_predict(X_ng_new)
    dbcvs.append(DBCV(X_ng_new, cluster_labels, dist_function=euclidean))

plt.plot(epss, dbcvs, '-o')
plt.xlabel(r'$\epsilon$')
plt.ylabel('DBCV');


opt_params_ng = [0.01, 0.03]

for i in opt_params_ng:
    dbscan = DBSCAN(eps=i, min_samples=7)
    cluster_labels = dbscan.fit_predict(X_ng_new)
    plt.scatter(ng_X, ng_Y, c=cluster_labels, s=0.6)
    plt.title(f'Epsilon = {i}')
    plt.show()
    print('Number of clusters:', cluster_labels.max()+1)
    print('Number of noise points:', (cluster_labels==-1).sum())
    print('Number of points:', len(cluster_labels))
    print('DBCV:', DBCV(X_ng_new, cluster_labels, dist_function=euclidean))