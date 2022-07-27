import collections
import time
from termcolor import colored
from data_validator.url_struct import URLStructure
from data_validator.vocab_struct import VocabStruct
from vectorizing.labse_embedder import SentenceEmbedder
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import json
from helpers.help_functions import just_non_zero_values_dict
from topic import TopicURL
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from pymongo import MongoClient
import warnings
import pandas as pd

from topic_vocab.vocab import cls_vocab
warnings.filterwarnings('ignore')


class WebSeparator:

    def __init__(
            self, embedder_conf_path='LaBSE',
            vocab_json_path='topic_vocab/vocab.json',
            from_db='mongo',
            print_feedback_tokens=False,
            device='cuda',
            distance_metric='cosine',
            connect_params=None,
            transform_value=0.5,
            threshold_other_value=0.35
    ):

        self.from_db = from_db
        self.print_feedback_tokens = print_feedback_tokens
        self.vocab_json_path = vocab_json_path
        self.embedder_conf_path = embedder_conf_path
        self.device_name = device
        self.distance_metric = distance_metric

        self.transform_value = transform_value
        self.threshold_other_value = threshold_other_value

        if connect_params is None:
            self.connect_params = {
                'connection_string': 'mongodb://yd-opslh001:27017',
                'db_name': 'hosts',
                'collection_name': 'hostsdata'
            }
        else:
            self.connect_params = connect_params

        self.nearest_clf = NearestCentroid(metric=self.distance_metric)
        self.agg_clusterer = AgglomerativeClustering(
            linkage='average',
            affinity=self.distance_metric
        )

        self.collection = self._connect_mongo()

        self.classes_vocab = self._build_vocab()

        self.embedder = SentenceEmbedder(
            device=self.device_name,
            path_enc_configs=self.embedder_conf_path
        )

        self.vocab_struct = VocabStruct(self.classes_vocab, metric=self.distance_metric)
        print(colored(f'Building vocab classes data structure', 'green'))
        self.vocab_struct.fill_embed_vocab(self.embedder)
        self.vocab_struct.fill_centroid_vocab()

        self.separator = TopicURL(vocab_embeddings=self.vocab_struct.embedded_vocab)
        
        print(self.vocab_struct.embedded_vocab)


    def _connect_mongo(self):
        client = MongoClient(self.connect_params['connection_string'])
        db = client[self.connect_params['db_name']]
        return db[self.connect_params['collection_name']]

    def _build_vocab(self):
        print(colored(f'Loading vocab classes data', 'green'))
        with open(self.vocab_json_path, 'r') as json_file:
            return json.load(json_file)

    def run(self, iterations=300, cls_vocab=cls_vocab):
        predicted_topics = collections.OrderedDict()
        print(colored(f'Running iterations', 'green'))
        start_time = time.time()

        topic_counter = collections.defaultdict(int)
        struct_json = collections.OrderedDict()

        for iteration, item in tqdm(enumerate(self.collection.find())):
            try:
                # ID = item['_id']
                url = item['Host']
                keywords = item['Keywords'].split(',')
                description = sent_tokenize(item['Description'])
                title = sent_tokenize(item['Title'])
                content = sent_tokenize(item['Content'])

                url_struct = URLStructure(
                    url_path=url,
                    keywords=keywords,
                    description=description,
                    title=title,
                    content=None,
                    fill_dict=True,
                    print_feedback_tokens=self.print_feedback_tokens
                )
                url_struct.fill_embed_fields(embedder=self.embedder)
                url_struct.form_labels_centroid_maintokens(
                    agg_clusterer=self.agg_clusterer,
                    centroid_clf=self.nearest_clf)
                url_struct.form_output_embeddings(
                    agg_clusterer=self.agg_clusterer)

                if self.print_feedback_tokens:
                    print(f'Feedback tokens : {url_struct.output_feedback_tokens}')
                    
                output_embeddings = url_struct.output_summary_embeddings

                distances_vocab = just_non_zero_values_dict(
                    self.separator.run(
                        output_embeddings,
                        b_value=self.transform_value,
                        threshold_other=self.threshold_other_value
                    ))

                # print(f'distances vocab : {distances_vocab}')
                
                

                pred_topic = list(distances_vocab.keys())
                print(f'Predicted topics : {pred_topic}\n\n')
                predicted_topics[url] = pred_topic


                struct_json[url] = {
                    'all_tokens': {
                        'keywords': item['Keywords'],
                        'description': item['Description'],
                        'title': item['Title'],
                        'content': item['Content']
                    },
                    'feedback_tokens': url_struct.output_feedback_tokens,
                    'predicted_topics_and_probs': distances_vocab
                }
                

                if iteration >= iterations:
                    break
            except:
                print(f'output feedback error')
                continue
            

        spended_time = round(time.time() - start_time, 2)
        print(colored(f'Finish ///', 'green'), colored(f'spended time : {spended_time}', 'red'))
        print(colored(f'Average time on website : {round(spended_time / iterations, 2)}', 'cyan'))
        print(colored(f'Average ws by one second: {int(1 / (spended_time / iterations))}', 'cyan'))

        with open('urls.json', 'w') as json_file:
            json.dump(struct_json, json_file)


if __name__ == '__main__':

    web_separator = WebSeparator(transform_value=1.1, print_feedback_tokens=True)
    web_separator.run(iterations=300)

    exit()
    with open('urls.json', 'r', encoding='UTF-8') as json_file:
        out_json_struct = json.load(json_file)

    for url in out_json_struct:
        print(f"{url} : {out_json_struct[url]['all_tokens']['title']}")
