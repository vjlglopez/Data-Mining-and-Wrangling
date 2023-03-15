data_wine = load_wine()
standard_scaler = StandardScaler()
X_wine = standard_scaler.fit_transform(data_wine['data'])
target_wine = data_wine['target']


X_wine_new = PCA(n_components=2, random_state=1337).fit_transform(X_wine)
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=target_wine);


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


X_ng_new = (TruncatedSVD(n_components=2, random_state=1337)
                .fit_transform(bow_ng))
plt.scatter(X_ng_new[:,0], X_ng_new[:,1], c=target_ng, alpha=0.8);


from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=5, linkage='single')
y_predict_wine = agg.fit_predict(X_wine)
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);


from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X_wine, method='single', optimal_ordering=True)


fig, ax = plt.subplots(figsize=(12,4), dpi=300)
dn = dendrogram(Z, ax=ax)
ax.set_ylabel(r'$\Delta$');


def plot1(Z):
    fig, ax = plt.subplots()
    dn = dendrogram(Z, ax=ax, p=5, truncate_mode='level')
    ax.set_ylabel(r'$\Delta$');
    return ax


ax = plot1(Z)
ax.figure.savefig('plot1-test.png')


from scipy.cluster.hierarchy import fcluster
y_predict_wine = fcluster(Z, t=3, criterion='distance')
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);

# single-linkage agglomerative clustering
agg = AgglomerativeClustering(n_clusters=None, linkage='single',
                              distance_threshold=3)
y_predict_wine = agg.fit_predict(X_wine)
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);


fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.flatten()
for i in range(1, 10):
    agg_ng = AgglomerativeClustering(n_clusters=i, linkage='single')
    y_predict_ng = agg_ng.fit_predict(bow_ng.toarray())
    ax[i-1].set_title(f'Single linkage clustering k = {i}')
    ax[i-1].scatter(X_ng_new[:,0], X_ng_new[:,1], c=y_predict_ng)
    fig.show()
    
    
Z = linkage(X_wine, method='complete', optimal_ordering=True)
fig, ax = plt.subplots()
dn = dendrogram(Z, ax=ax)
ax.set_ylabel(r'$h$');


plot1(Z);


from scipy.cluster.hierarchy import fcluster
y_predict_wine = fcluster(Z, t=9, criterion='distance')
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);

# complete-linkage agglomerative clustering
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.flatten()
for i in range(1, 10):
    agg_ng = AgglomerativeClustering(n_clusters=i, linkage='complete')
    y_predict_ng = agg_ng.fit_predict(bow_ng.toarray())
    ax[i-1].set_title(f'Complete linkage clustering k = {i}')
    ax[i-1].scatter(X_ng_new[:,0], X_ng_new[:,1], c=y_predict_ng)
    fig.show()
    
    
Z = linkage(X_wine, method='average', optimal_ordering=True)
fig, ax = plt.subplots()
dn = dendrogram(Z, color_threshold=4.8, ax=ax)
ax.set_ylabel(r'$h$');


plot1(Z);


from scipy.cluster.hierarchy import fcluster
y_predict_wine = fcluster(Z, t=4.75, criterion='distance')
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);

# average-linkage agglomerative clustering
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.flatten()
for i in range(1, 10):
    agg_ng = AgglomerativeClustering(n_clusters=i, linkage='average')
    y_predict_ng = agg_ng.fit_predict(bow_ng.toarray())
    ax[i-1].set_title(f'Average linkage clustering k = {i}')
    ax[i-1].scatter(X_ng_new[:,0], X_ng_new[:,1], c=y_predict_ng)
    fig.show()
    
    
Z = linkage(X_wine, method='ward', optimal_ordering=True)
fig, ax = plt.subplots()
dn = dendrogram(Z, ax=ax)
ax.set_ylabel(r'$h$');


plot1(Z);


from scipy.cluster.hierarchy import fcluster
y_predict_wine = fcluster(Z, t=15, criterion='distance')
plt.scatter(X_wine_new[:,0], X_wine_new[:,1], c=y_predict_wine);

# ward-linkage agglomerative clustering
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.flatten()
for i in range(1, 10):
    agg_ng = AgglomerativeClustering(n_clusters=i, linkage='ward')
    y_predict_ng = agg_ng.fit_predict(bow_ng.toarray())
    ax[i-1].set_title(f"Ward's method clustering k = {i}")
    ax[i-1].scatter(X_ng_new[:,0], X_ng_new[:,1], c=y_predict_ng)
    fig.show()