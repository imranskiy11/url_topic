from typing import List
import collections
from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import numpy as np
from termcolor import colored

def show_clustering_labels(data, labels):
    res = collections.defaultdict(list)
    for i in range(len(labels)):
        label_num = labels[i] + 1
        current_token = data[i]
        res[f'cluster::{label_num}'].append(current_token)
    for cluster_num in dict(sorted(res.items(), key=lambda x: x[0])):
            print(f'{cluster_num} : {res[cluster_num]}')
            print('-----'*20)
   

def agglomerative_labels_and_centroids(embeddings, agg_clusterer, centroid_clf, data=None, verbose=0):
    #clustering
    if len(embeddings) == 1:
        return [0], embeddings
    agg_predict = agg_clusterer.fit_predict(embeddings)
    labels = agg_clusterer.labels_
    if verbose == 1: 
        print(f'Clusters: {labels}')
        show_clustering_labels(data, labels)
    #get centroid        
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

def values_from_key_with_many_values(labels_dict: dict):
    curr_max = 0
    out_key = list()
    for key in labels_dict:
        if curr_max <= len(list(labels_dict[key])):
            curr_max = len(list(labels_dict[key]))
            out_key.append(labels_dict[key])
    return flatten(out_key)

class URLStructure:

    def __init__(
                self,
                url_path: str,
                keywords: List[str]=None,
                title: List[str]=None, 
                description: List[str]=None,
                content: List[str]=None,
                embedded_keywords = None, 
                embedded_title = None,
                embedded_description = None,
                embedded_content = None,
                fill_dict=False,
                ):
                
        self.url_path = url_path
        
        
        self.keywords = keywords
        self.title = title
        self.description = description
        self.content = content
          
        self.embedded_keywords = embedded_keywords
        self.embedded_title = embedded_title
        self.embedded_description = embedded_description
        self.embedded_content = embedded_content


        self.is_keywords_not_null_or_empty = bool(self.keywords)
        self.is_title_not_null_or_empty = bool(self.title)
        self.is_description_not_null_or_empty = bool(self.description)
        self.is_content_not_null_or_empty = bool(self.content)
        
        if fill_dict:
            self.filled_dict = collections.OrderedDict()
            self.fill_dict()
            # print(f'inputs: {self.fields_name}')
            
        
    # def fill_embed_fields(self, embedder):
    #     for tokens_input, embedded_token_input in zip(
    #         [self.keywords, self.title, self.description, self.content],
    #         [self.embedded_keywords, self.embedded_title, self.embedded_description, self.embedded_content]):
        
    #         if bool(tokens_input):
    #             print(bool(tokens_input))
    #             embedded_token_input = embedder.word_embeddings_list(
    #                 tokens_input).detach().cpu().detach() 
                    

    def fill_embed_fields(self, embedder):
        if self.is_keywords_not_null_or_empty:
            self.embedded_keywords = embedder.word_embeddings_list(
                self.keywords).detach().cpu().detach()      
        if self.is_title_not_null_or_empty:
            self.embedded_title = embedder.word_embeddings_list(
                self.title).detach().cpu().detach()             
        if self.is_description_not_null_or_empty:
            self.embedded_description = embedder.word_embeddings_list(
                self.description).detach().cpu().detach()
        if self.is_content_not_null_or_empty:
            self.embedded_content = embedder.word_embeddings_list(
                self.content).detach().cpu().detach()
    
    def fill_dict(self):
        if self.is_keywords_not_null_or_empty:
            self.filled_dict['keywords'] = self.keywords
        if self.is_description_not_null_or_empty:
            self.filled_dict['description'] = self.description
        if self.is_title_not_null_or_empty:
            self.filled_dict['title'] = self.title
        if self.is_content_not_null_or_empty:
                self.filled_dict['content'] = self.content
    
    @property
    def fields_name(self):
        return list(self.filled_dict.keys())
 
    def form_labels_centroid_maintokens(self, agg_clusterer, centroid_clf, verbose=0, packing=True, save_feedback_tokens=False):
        if self.embedded_keywords is not None:
            self.keywords_labels, self.keywords_centroids = \
            agglomerative_labels_and_centroids(
                embeddings=self.embedded_keywords,
                agg_clusterer=agg_clusterer,
                centroid_clf=centroid_clf,
                data=self.keywords,
                verbose=verbose)
            self.keywords_main_tokens_dict = nearest_embeddings(
                embeddings=self.embedded_keywords,
                centroids=self.keywords_centroids,
                labels=self.keywords_labels
            )               
        if self.embedded_description is not None:     
            self.description_labels, self.description_centroids = \
            agglomerative_labels_and_centroids(
                embeddings=self.embedded_description,
                agg_clusterer=agg_clusterer,
                centroid_clf=centroid_clf,
                data=self.description,
                verbose=verbose)
            self.description_main_tokens_dict = nearest_embeddings(
                embeddings=self.embedded_description,
                centroids=self.description_centroids,
                labels=self.description_labels
            )              
        if self.embedded_title is not None:        
            self.title_labels, self.title_centroids = \
            agglomerative_labels_and_centroids(
                embeddings=self.embedded_title,
                agg_clusterer=agg_clusterer,
                centroid_clf=centroid_clf,
                data=self.title,
                verbose=verbose)
            self.title_main_tokens_dict = nearest_embeddings(
                embeddings=self.embedded_title,
                centroids=self.title_centroids,
                labels=self.title_labels
            )          
        if self.embedded_content is not None:
            self.content_labels, self.content_centroids = \
            agglomerative_labels_and_centroids(
                embeddings=self.embedded_content,
                agg_clusterer=agg_clusterer,
                centroid_clf=centroid_clf,
                data=self.content,
                verbose=verbose)
            self.content_main_tokens_dict = nearest_embeddings(
                embeddings=self.embedded_content,
                centroids=self.content_centroids,
                labels=self.content_labels
            )
        self.pack_modality_centroids(save_tokens_feedback=save_feedback_tokens)
        # print(f'all modality shape: {self.all_modulity_embeddings.shape}')
        # print(f'feedback tokens: {self.all_modality_feedback_tokens}')
               
    def _generate_by_idxs(self, iter_data, idxs):
        for idx in idxs:
            yield iter_data[idx]
      
    def _main_tokens(self, embeddings, tokens_dict, data=None, embedded=False):
        if embedded:
            return  np.vstack(
                [embed_token for embed_token in self._generate_by_idxs(
                    embeddings, list(tokens_dict.values())
                )])
        if data is not None:
            return [token for token in self._generate_by_idxs(
                                                data, list(tokens_dict.values())
                                                )]
        else:
            raise Exception(colored('Not passed data', 'red'))
                                                        
    def keywords_main_tokens(self, embedded=False):
        if self.is_keywords_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_keywords,
                data=self.keywords,
                tokens_dict=self.keywords_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nKeywords was passed an empty list or null\n', 'red'))
            print(colored('Keywords was passed an empty list or null', 'red'))
            return None
        
    def title_main_tokens(self, embedded=False):
        if self.is_title_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_title,
                data=self.title,
                tokens_dict=self.title_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nTitle was passed an empty list or null\n', 'red'))
            print(colored('Title was passed an empty list or null', 'red'))
            return None
        
    def description_main_tokens(self, embedded=False):
        if self.is_description_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_description,
                data=self.description,
                tokens_dict=self.description_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nDescription was passed an empty list or null\n', 'red'))
            print(colored('Description was passed an empty list or null', 'red'))
            return None
        
    def content_main_tokens(self, embedded=False):
        if self.is_content_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_content,
                data=self.content,
                tokens_dict=self.content_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nContents data was passed an empty list or null\n', 'red'))
            print(colored('Content was passed an empty list or null', 'red'))
            return None

    def pack_modality_centroids(self, save_tokens_feedback=False):
        all_modality = [
                self.keywords_main_tokens(embedded=True),
                self.title_main_tokens(embedded=True),
                self.description_main_tokens(embedded=True),
                self.content_main_tokens(embedded=True)
            ]
        
        self.all_modulity_embeddings = np.vstack(
            list(
                filter(lambda el: el is not None, all_modality)
            )
        )
        
        if save_tokens_feedback:
            feedback_tokens = [
                self.keywords_main_tokens(embedded=False),
                self.title_main_tokens(embedded=False),
                self.description_main_tokens(embedded=False),
                self.content_main_tokens(embedded=False)
            ]
            self.all_modality_feedback_tokens = flatten(
                list(
                    filter(lambda el: el is not None, feedback_tokens)
                )
            )
            
    def form_output_embeddings(self, agg_clusterer, verbose=0, save_feedback_tokens=False):
        if len(self.all_modulity_embeddings) == 1:
            return self.all_modulity_embeddings
        agg_clusterer.fit(self.all_modulity_embeddings)
        valid_idxs = values_from_key_with_many_values(get_labels_by_clusters(agg_clusterer.labels_))
        if verbose == 1: 
            print(f'Clusters: {agg_clusterer.labels_}')
            print(f'labels :{get_labels_by_clusters(agg_clusterer.labels_)}')
            print(f'valid indexes : {valid_idxs}')
        self.output_summary_embeddings = \
            np.vstack([embedding for embedding in self._generate_by_idxs(
                        self.all_modulity_embeddings, valid_idxs)])
        if save_feedback_tokens:
            self.output_feedback_tokens = [token for token in self._generate_by_idxs(
                self.all_modality_feedback_tokens, valid_idxs
            )]