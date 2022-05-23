import collections
from pandas import Series

def just_non_zero_values_dict(input_dict):
    return {key:value for key, value in input_dict.items() if value > 0 }
    
def flatten(t):
    return [item for sublist in t for item in sublist]
    
def get_labels_by_clusters(labels):
    clusters_labels_dict = collections.OrderedDict()
    pd_labels = Series(labels)
    unique_labels = tuple(pd_labels.unique())
    for label in unique_labels:
        clusters_labels_dict[label] = list(pd_labels[pd_labels == label].index)
    return clusters_labels_dict
    
def cluster_embeddings_generator(embeddings_array, indexes):
    for idx in indexes:
        yield embeddings_array[idx]
