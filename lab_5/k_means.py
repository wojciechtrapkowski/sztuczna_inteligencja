import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids_indices = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[centroids_indices]
    print(centroids)
    return centroids

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization

    # Select first centroid
    centroids = [data[np.random.randint(data.shape[0])]]
    
    for _ in range(1, k):
        distances = []
        for v in data:
            distances.append(np.min([euclidean_distance(v, centroid) for centroid in centroids]))
    
        new_centroid = data[np.argmax(distances)]
        centroids.append(new_centroid)
    
    return np.array(centroids)

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    assignments = np.zeros(data.shape[0], dtype=int)

    for i, point in enumerate(data):
        min_distance = float('inf')
        nearest_centroid_idx = -1

        for j, cen in enumerate(centroid):
            distance = euclidean_distance(point, cen)
            if distance < min_distance:
                min_distance = distance
                nearest_centroid_idx = j
        
        assignments[i] = nearest_centroid_idx

    return assignments

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    num_clusters = len(np.unique(assignments))
    new_centroids = np.zeros((num_clusters, data.shape[1]))

    for cluster in range(num_clusters):
        cluster_points = data[assignments == cluster]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids[cluster] = new_centroid
    return new_centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments  = assign_to_cluster(data, centroids)

    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)      

