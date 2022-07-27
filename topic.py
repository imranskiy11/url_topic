from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import collections
from termcolor import colored
from sentence_transformers import util

class OldTopicURL:
    
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
        
    def cosine_similarity_matrix(self, vecs1, vecs2, cosine_dist_threshold=0, round_val=4):
        return cosine_similarity(vecs1, vecs2).flatten()  
        # cosine_sim = cosine_similarity(vecs1, vecs2).flatten()        
        # return cosine_sim[cosine_sim >= cosine_dist_threshold]
    
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
    
    

    def run(self, embeddings, b_value, threshold_other, transform=False):
        curr_dict = self.form_distances_vocab(embeddings, threshold_other=threshold_other)
        if not transform:
            return{ k:v for k, v in curr_dict.items() if v >= 0.37}
        else:
            return self.transform_dict(curr_dict, b_value)
            
            
class TopicURL:
    
    def __init__(self, vocab_embeddings, distance_metric='cosine'):
        super(TopicURL, self).__init__()
        
        self.distance_metric = distance_metric
        self.softmax_layer = torch.nn.Softmax(dim=0)
        self.vocab_embeddings = vocab_embeddings
        
    @staticmethod
    def get_scores_from_semantic_search(hits: list):
        current_max = 0
        for i in hits:
            scores = list()
            means = list()
            for j in i:
            #     if current_max <= j['score']:
            #         current_max = j['score']
            # scores.append(current_max)
                means.append(j['score'])
            scores.append(np.max(np.array(means)))
        return np.around(np.array(scores), 3)
        
    
    def _distance(self, emb_left, emb_right, round_val=4):
        if self.distance_metric == 'cosine':
            return round(1 - cosine(emb_left, emb_right), round_val)
        if self.distance_metric == 'euclidean':
            return round(euclidean(emb_left, emb_right), round_val)
            
    def _transform_list(self, x: list, b=1)-> list:
        return [round(v**(4*b*(i**2)), 3) for i, v in enumerate(x)]
        
    def get_mult_tensor(self, l: list, mult_value: float=10):
        return np.array([10 if i > 0 else 0 for i in l])
        
    def cosine_similarity_matrix(self, vecs1, vecs2, cosine_dist_threshold=0, round_val=4):
        # print(f'sim matrix : {cosine_similarity(vecs1, vecs2).flatten() }')
        return cosine_similarity(vecs1, vecs2).flatten()  
        # cosine_sim = cosine_similarity(vecs1, vecs2).flatten()        
        # return cosine_sim[cosine_sim >= cosine_dist_threshold]
        
    
    def form_distances_vocab(self, embeddings, threshold_other=0.3, round_val=4, calc_type='scores'):
        # if not bool(embeddings):
        #     raise Exception(colored('embeddings is None', 'red'))
        vocab_distances = collections.OrderedDict()
        for class_name in self.vocab_embeddings:
            if calc_type == 'scores':
                vocab_distances[class_name] = self.get_scores_from_semantic_search(
                    util.semantic_search(self.vocab_embeddings[class_name], embeddings))
            else:
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
    
    

    def run(self, embeddings, b_value, threshold_other, transform=False):
        curr_dict = self.form_distances_vocab(embeddings, threshold_other=threshold_other)
        if not transform:
            return{ k:v for k, v in curr_dict.items() if v >= 0.37}
        else:
            return self.transform_dict(curr_dict, b_value)
    