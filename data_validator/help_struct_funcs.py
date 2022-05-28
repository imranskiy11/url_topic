import collections
import pandas as pd
from scipy.spatial.distance import cosine


def show_clustering_labels(data, labels):
    res = collections.defaultdict(list)
    for i in range(len(labels)):
        label_num = labels[i] + 1
        current_token = data[i]
        res[f'cluster::{label_num}'].append(current_token)
    for cluster_num in dict(sorted(res.items(), key=lambda x: x[0])):
        print(f'{cluster_num} : {res[cluster_num]}')
        print('-----' * 20)


def agglomerative_labels_and_centroids(embeddings, agg_clusterer, centroid_clf, more_info=False, data=None):
    # clustering
    if len(embeddings) == 1:
        return [0], embeddings
    agg_predict = agg_clusterer.fit_predict(embeddings)
    labels = agg_clusterer.labels_
    if more_info:
        assert data is not None, 'not passed data (None)'
        print(f'Clusters: {labels}')
        show_clustering_labels(data, labels)  # add data from func inputs
    # get centroid
    centroid_clf.fit(embeddings, agg_predict)
    centroids = centroid_clf.centroids_
    return labels, centroids


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_labels_by_clusters(labels):
    clusters_labels_dict = collections.OrderedDict()
    pd_labels = pd.Series(labels)
    unique_labels = tuple(pd_labels.unique())
    for label in unique_labels:
        clusters_labels_dict[label] = list(pd_labels[pd_labels == label].index)
    return clusters_labels_dict


def cluster_embeddings_generator(embeddings_array, indexes):
    for idx in indexes:
        yield embeddings_array[idx]


def nearest_embeddings(embeddings, centroids, labels, metric='cosine'):
    nearest_dict = collections.OrderedDict()
    cluster_labels_dict = get_labels_by_clusters(labels)
    if metric == 'cosine':
        for num_centroid, centroid in enumerate(centroids):
            current_clusters_embeddings = [emb for emb in cluster_embeddings_generator(
                embeddings, cluster_labels_dict[num_centroid])]
            current_distance = 0

            for num_embedding, embedding in enumerate(current_clusters_embeddings):
                dist = 1 - cosine(centroid, embedding)
                if current_distance < dist:
                    current_distance = dist
                    nearest_dict[f'{num_centroid}_centroid'] = cluster_labels_dict[num_centroid][num_embedding]
    return nearest_dict


def values_from_key_with_maxlen_values(labels_dict: dict):
    curr_max = 0
    out_key = list()
    for key in labels_dict:
        if curr_max <= len(list(labels_dict[key])):
            curr_max = len(list(labels_dict[key]))
            out_key.append(labels_dict[key])
    return flatten(out_key)
