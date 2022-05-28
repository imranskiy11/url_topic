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

    def form_labels_centroid_maintokens(self, agg_clusterer, centroid_clf):
        if self.embedded_keywords is not None:
            self.keywords_labels, self.keywords_centroids = \
                agglomerative_labels_and_centroids(
                    embeddings=self.embedded_keywords,
                    agg_clusterer=agg_clusterer,
                    centroid_clf=centroid_clf,
                    data=self.keywords)
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
                    data=self.description)
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
                    data=self.title)
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
                    data=self.content)
            self.content_main_tokens_dict = nearest_embeddings(
                embeddings=self.embedded_content,
                centroids=self.content_centroids,
                labels=self.content_labels
            )
        self.pack_modality_centroids()

    @staticmethod
    def _generate_by_idxs(iter_data, idxs):
        for idx in idxs:
            yield iter_data[idx]

    def _main_tokens(self, embeddings, tokens_dict, data=None, embedded=False):
        if embedded:
            return np.vstack(
                [embed_token for embed_token in self._generate_by_idxs(
                    embeddings, list(tokens_dict.values())
                )])
        if data is not None:
            return [token for token in self._generate_by_idxs(
                data, list(tokens_dict.values())
            )]
        else:
            raise Exception(colored('Not passed data', 'red'))

    def keywords_main_tokens(self, embedded=False, info=False):
        if self.is_keywords_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_keywords,
                data=self.keywords,
                tokens_dict=self.keywords_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nKeywords was passed an empty list or null\n', 'red'))
            if info:
                print(colored('Keywords was passed an empty list or null', 'red'))
            return None

    def title_main_tokens(self, embedded=False, info=False):
        if self.is_title_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_title,
                data=self.title,
                tokens_dict=self.title_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nTitle was passed an empty list or null\n', 'red'))
            if info:
                print(colored('Title was passed an empty list or null', 'red'))
            return None

    def description_main_tokens(self, embedded=False, info=False):
        if self.is_description_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_description,
                data=self.description,
                tokens_dict=self.description_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nDescription was passed an empty list or null\n', 'red'))
            if info:
                print(colored('Description was passed an empty list or null', 'red'))
            return None

    def content_main_tokens(self, embedded=False, info=False):
        if self.is_content_not_null_or_empty:
            return self._main_tokens(
                embeddings=self.embedded_content,
                data=self.content,
                tokens_dict=self.content_main_tokens_dict,
                embedded=embedded
            )
        else:
            # raise Exception(colored('\nContents data was passed an empty list or null\n', 'red'))
            if info:
                print(colored('Content was passed an empty list or null', 'red'))
            return None

    def pack_modality_centroids(self):
        all_modality = [
            self.keywords_main_tokens(embedded=True),
            self.title_main_tokens(embedded=True),
            self.description_main_tokens(embedded=True),
            self.content_main_tokens(embedded=True)
        ]

        self.all_modality_embeddings = np.vstack(
            list(
                filter(lambda el: el is not None, all_modality)
            )
        )

        if self.print_feedback_tokens:
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

    def form_output_embeddings(self, agg_clusterer, more_info=False):
        if len(self.all_modality_embeddings) == 1:
            return self.all_modality_embeddings
        agg_clusterer.fit(self.all_modality_embeddings)
        valid_idxs = values_from_key_with_maxlen_values(get_labels_by_clusters(agg_clusterer.labels_))
        if more_info:
            print(f'Clusters: {agg_clusterer.labels_}')
            print(f'labels :{get_labels_by_clusters(agg_clusterer.labels_)}')
            print(f'valid indexes : {valid_idxs}')
        self.output_summary_embeddings = \
            np.vstack([embedding for embedding in self._generate_by_idxs(
                self.all_modality_embeddings, valid_idxs)])
        if self.print_feedback_tokens:
            try:
                self.output_feedback_tokens = [token for token in self._generate_by_idxs(
                    self.all_modality_feedback_tokens, valid_idxs
                )]
            except:
                self.output_feedback_tokens = list()
