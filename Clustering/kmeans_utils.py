from sklearn.cluster import KMeans

def compute_wcss(X, max_k=10):
    """حساب WCSS لاستخدام Elbow Method"""
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(  
            n_clusters=i,  
            init='k-means++',  
            n_init=20,
            algorithm='elkan',
            tol=0.001,
            random_state=42  
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def apply_kmeans(X, n_clusters=3):
    """تطبيق KMeans وإرجاع الكائن والتصنيفات"""
    kmeans = KMeans(  
        n_clusters=n_clusters,  
        init='k-means++',  
        n_init=20,
        algorithm='elkan',
        tol=0.001,
        random_state=42
    )
    y_pred = kmeans.fit_predict(X)
    return kmeans, y_pred
