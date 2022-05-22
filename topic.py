from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import collections

class TopicURL:
    
    def __init__(self, vocab_embeddings, distance_metric='cosine'):
        super(TopicURL, self).__init__()
        
        self.distance_metric = distance_metric
        self.softmax = torch.nn.Softmax(dim=0)
        self.vocab_embeddings = vocab_embeddings
        
    
    def _distance(self, emb_left, emb_right, round_val=4):
        if self.distance_metric == 'cosine':
            return round(1 - cosine(emb_left, emb_right), round_val)
        if self.distance_metric == 'euclidean':
            return round(euclidean(emb_left, emb_right), round_val)

    def cosine_similarity_matrix(self, vecs1, vecs2, round_val=4):
        # print(f'left shape : {vecs1.shape}\nright shape : {vecs2.shape}')
        return cosine_similarity(vecs1, vecs2).flatten() #.round(round_val)        
    
    
    def form_distances_vocab(self, embeddings, round_val=4):
        vocab_distances = collections.OrderedDict()
        for class_name in self.vocab_embeddings:
            vocab_distances[class_name] = np.median(
                self.cosine_similarity_matrix(
                                embeddings, self.vocab_embeddings[class_name]
                                )
            )
        return dict(sorted(vocab_distances.items(), key=lambda x: x[1], reverse=True))
    

        
        
if __name__ == '__main__':
    pass
    
    
    