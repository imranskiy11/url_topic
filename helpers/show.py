import collections

def show_clustering_labels(data, labels):
    res = collections.defaultdict(list)
    for i in range(len(labels)):
        label_num = labels[i] + 1
        current_token = data[i]
        res[f'cluster::{label_num}'].append(current_token)
    for cluster_num in dict(sorted(res.items(), key=lambda x: x[0])):
            print(f'{cluster_num} : {res[cluster_num]}')
            print('-----'*20)

