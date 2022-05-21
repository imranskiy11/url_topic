import collections
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
from termcolor import colored

class VocabStruct:
    
    def __init__(self, vocab_classes_dict, embedded_vocab_dict=None, metric='cosine'):
        self.vocab = vocab_classes_dict
        self.metric = metric
        
        if embedded_vocab_dict is None:
            self.embedded_vocab = collections.OrderedDict()
            self.centroids_vocab = collections.OrderedDict()
        
        
        self.centroid_clf = NearestCentroid(metric=self.metric)
        self.agg_cluster = AgglomerativeClustering(affinity='cosine', linkage='average')
        
    
    def fill_embed_vocab(self, embedder):
        for class_name in self.vocab:
            self.embedded_vocab[class_name] = embedder.word_embeddings_list(
                self.vocab[class_name]).detach().cpu().detach()
                
    def fill_centroid_vocab(self):
        if len(self.embedded_vocab) == 0:
            raise Exception(colored('Classes embedded vocab is empty', 'red'))
        for class_name in self.embedded_vocab:
            agg_preds = self.agg_cluster.fit_predict(self.embedded_vocab[class_name])
            self.centroid_clf.fit(self.embedded_vocab[class_name], agg_preds)
            self.centroids_vocab[class_name] = self.centroid_clf.centroids_
            
