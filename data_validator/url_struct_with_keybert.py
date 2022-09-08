from typing import List
import collections
from .help_struct_funcs import agglomerative_labels_and_centroids, nearest_embeddings
from .help_struct_funcs import get_labels_by_clusters, flatten, values_from_key_with_maxlen_values
import numpy as np
from termcolor import colored


class URLStructure:
    def __init__(
            self,
            url_path: str,
            keywords: List[str] = None,
            title: List[str] = None,
            description: List[str] = None,
            content: List[str] = None,
            embedded_keywords=None,
            embedded_title=None,
            embedded_description=None,
            embedded_content=None,
            fill_dict=False,
            print_feedback_tokens=False
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
        
        self.print_feedback_tokens = print_feedback_tokens
        
        if fill_dict:
            self.filled_dict = collections.OrderedDict()
            self.fill_dict()
            
            
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
        


if __name__ == '__main__':
    pass