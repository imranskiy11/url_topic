from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import collections
from termcolor import colored

class TopicURL:
    
    def __init__(self, vocab_embeddings, distance_metric='cosine'):
        super(TopicURL, self).__init__()
        
        self.distance_metric = distance_metric
        self.softmax_layer = torch.nn.Softmax(dim=0)
        self.vocab_embeddings = vocab_embeddings
        
    
    def _distance(self, emb_left, emb_right, round_val=4):
        if self.distance_metric == 'cosine':
            return round(1 - cosine(emb_left, emb_right), round_val)
        if self.distance_metric == 'euclidean':
            return round(euclidean(emb_left, emb_right), round_val)
            
    def _transform_list(self, x: list, b=1)-> list:
        return [round(v**(4*b*(i**2)), 3) for i, v in enumerate(x)]
        
    def get_mult_tensor(self, l: list, mult_value: float=10):
        return np.array([10 if i > 0 else 0 for i in l])
        
    def cosine_similarity_matrix(self, vecs1, vecs2, round_val=4):
        return cosine_similarity(vecs1, vecs2).flatten()    
    
    def form_distances_vocab(self, embeddings, threshold_other=0.3, round_val=4):
        # if not bool(embeddings):
        #     raise Exception(colored('embeddings is None', 'red'))
        vocab_distances = collections.OrderedDict()
        for class_name in self.vocab_embeddings:
            vocab_distances[class_name] = np.median(
                self.cosine_similarity_matrix(
                    embeddings, self.vocab_embeddings[class_name]))
        if max(list(vocab_distances.values())) <= threshold_other:
            return {'Другое': 1.0}
        return dict(sorted(vocab_distances.items(), key=lambda x: x[1], reverse=True))
        
    def transform_dict(self, input_dict: dict, b_value=1, round_value=4):
        class_names = list(input_dict.keys())
        transformed_values = self._transform_list(list(input_dict.values()), b=b_value)
        multiply_tensor = self.get_mult_tensor(transformed_values)
        non_zero_len = len(multiply_tensor[multiply_tensor != 0])
        softmax_layer_output = self.softmax_layer(
            torch.from_numpy(np.array(transformed_values)[:non_zero_len]))
        softmax_layer_output = list(
            torch.cat(
                (softmax_layer_output, torch.zeros(multiply_tensor.shape[0] - non_zero_len))
            ).numpy())
        return {class_names[i]: round(softmax_layer_output[i], round_value) for i in range(len(softmax_layer_output))}
    
    
    def run(self, embeddings, b_value):
        return self.transform_dict(self.form_distances_vocab(embeddings), b_value)
        
if __name__ == '__main__':
    dataset1 = np.vstack(
        [
            np.array([1, 2, 3, 2]), np.array([4, 2, 3, 1.4]), np.array([1, 2, 3, 6])
        ]
    )
    dataset2 = np.vstack(
        [
            np.array([12, 2, 3, 2]), np.array([1, 3, 3, 1.4]), np.array([8, 2, 3, 1]),
            np.array([2, 4, 7, 2]), np.array([11, 13, 3, 2.1]), np.array([9, 12, 4, 1])
        ]
    )
    dataset3 = np.vstack(
        [
            np.array([1, 2, 3, 2]), np.array([1, 2, 3, 6]), np.array([1, 2, 3, 2]), np.array([1, 2, 3, 6])
        ]
    )
    
    # print(f'datasets shapes : {dataset1.shape}, {dataset3.shape}')
    
    # cos_sim = cosine_similarity(
    #     dataset1, dataset3
    # )
    # print(cos_sim.flatten().round(4))
    vocab_dict = collections.OrderedDict({
        'key1': dataset3
    })
    topic = TopicURL(vocab_dict)
    print(topic.cosine_similarity_matrix(dataset1, dataset3))
    
    
    