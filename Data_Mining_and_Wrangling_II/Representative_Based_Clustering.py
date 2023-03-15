def pooled_within_ssd(X, y, centroids, dist):
    score_list = []
    for i in range(len(centroids)):
        df = pd.DataFrame(X)
        df['labeled'] = y
        df = (
            df[df['labeled']==i]
            .drop(columns='labeled')
            .reset_index(drop=True)
        )
        ls = []
        for j in range(len(df)):
            ls.append(df.loc[j].tolist())
        score = 0
        for k in range(len(ls)):
            score += 1/(2*len(ls)) * dist(ls[k], centroids[i])**2
        score_list.append(score)
    return sum(score_list)


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    rng = np.random.default_rng(random_state)
    gap_stats = []
    w_k = pooled_within_ssd(X, y, centroids, dist)
    for _ in range(b):
        X_new = rng.uniform(low=np.min(X, axis=0),
                         high=np.max(X, axis=0),
                         size=X.shape)
        k_sorter = clusterer
        y_new = k_sorter.fit_predict(X_new)
        centroids_new = k_sorter.cluster_centers_
        w_ki = pooled_within_ssd(X_new, y_new, centroids_new, dist)
        gap_stats.append(np.log(w_ki) - np.log(w_k))
    gap_statistic = np.sum(gap_stats) / b
    gap_statistic_std = np.std(gap_stats)
    return gap_statistic, gap_statistic_std

def purity(y_true, y_pred):
    purity = (
        sum(confusion_matrix(y_true, y_pred).max(axis=0)) /
        len(y_true)
    )
    return purity


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k, random_state=1337)

        y = clusterer_k.fit_predict(X)
        ys.append(y)

        centers.append(clusterer_k.cluster_centers_)

        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X, y))
        scs.append(silhouette_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
        gs = gap_statistic(X, y, clusterer_k.cluster_centers_, 
                                 euclidean, 5, 
                                 clone(clusterer).set_params(n_clusters=k), 
                                 random_state=1337)
        gss.append(gs[0])
        gssds.append(gs[1])
        if actual is not None:
            ps.append(purity(actual, y))
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y))
    keys = ['ys', 'centers', 'inertias', 'chs',
            'scs', 'dbs', 'gss', 'gssds']
    if actual is not None:
        keys.extend(['ps', 'amis', 'ars'])
        dict_act_Not_None = (
            dict(zip(keys,
                     [ys, centers, inertias, chs, scs,
                      dbs, gss, gssds, ps, amis, ars]))
        )
        return dict_act_Not_None
    else:
        dict_act_None = (
            dict(zip(keys,
                     [ys, centers, inertias, chs, 
                      scs, dbs, gss, gssds]))
        )
        return dict_act_None


def plot_clusters(X, ys, centers, transformer):
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y,cs in zip(range(2, k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax


def plot_internal(inertias, chs, scs, dbs, gss, gssds):
    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt='-go', label='Gap statistic')
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.plot(ks, dbs, '-gs', label='DB')
    ax2.set_ylabel('Gap statistic/Silhouette/DB')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax


def plot_external(ps, amis, ars):
    fig, ax = plt.subplots()
    ks = np.arange(2, len(ps)+2)
    ax.plot(ks, ps, '-o', label='PS')
    ax.plot(ks, amis, '-ro', label='AMI')
    ax.plot(ks, ars, '-go', label='AR')
    ax.set_xlabel('$k$')
    ax.set_ylabel('PS/AMI/AR')
    ax.legend()
    return ax


def gap_statistic_kmedoids(X, y, centroids, b):
    w_ki_list = []
    w_k = pooled_within_ssd(X, y, centroids, euclidean)
    for i in range(b):
        X_new = np.random.uniform(low=np.min(X, axis=0),
                                     high=np.max(X, axis=0),
                                     size=X.shape)
        kmed = kmedoids(X_new, np.arange(len(centroids)), ccore=True)
        kmed.process()
        clusters = kmed.get_clusters()
        centers = kmed.get_medoids()
        y_new = np.zeros(len(X)).astype(int)
        for cluster, point in enumerate(clusters):
            y_new[point] = cluster
        c_centers = np.array(centers)
        w_ki = pooled_within_ssd(X_new, y_new, X_new[centers], euclidean)
        w_ki_list.append(np.log(w_ki))
    gap_stat_kmed = np.mean(w_ki_list) - np.log(w_k)
    gap_std_kmed = np.std(w_ki_list)
    return gap_stat_kmed, gap_std_kmed


def cluster_range_kmedoids(X, k_start, k_stop, actual=None):
    ys = []
    cs = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    X = np.asarray(X)
    for k in range(k_start, k_stop+1):
        clusterer_k = kmedoids(X, np.arange(k), ccore=True)
        
        clusterer_k.process()
        y = np.zeros(len(X)).astype(int)
        clusters = clusterer_k.get_clusters()
        for cluster, point in enumerate(clusters):
            y[point] = cluster
            
        centers = X[clusterer_k.get_medoids(), :]

        se = []
        for i, x in enumerate(X):
            se.append(euclidean(x, centers[y[i]])**2)
        sse = np.sum(se)
        
        gs = gap_statistic_kmedoids(X, y, centers, 5)
        gss.append(gs[0])
        gssds.append(gs[1])
        
        cs.append(centers)
        inertias.append(sse)
        ys.append(y)
        chs.append(calinski_harabasz_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
        scs.append(silhouette_score(X, y))
        
        if actual is not None:
            ps.append(purity(actual, y))
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y))
    
    if actual is not None:
        c_dict = {
            'ys': ys, 'centers': cs, 'inertias': inertias,
            'chs': chs, 'scs': scs, 'dbs': dbs, 'gss': gss,
            'gssds': gssds, 'ps': ps, 'amis': amis, 'ars': ars
        }
    else:
        c_dict = {
            'ys': ys, 'centers': cs, 'inertias': inertias, 'chs': chs,
            'scs': scs, 'dbs': dbs, 'gss': gss, 'gssds': gssds
    }
    return c_dict