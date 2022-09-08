import warnings
warnings.filterwarnings('ignore')

from data_validator.url_struct_with_keybert import URLStructure

import time
from termcolor import colored
import torch
from data_validator.url_struct import URLStructure
from data_validator.vocab_struct import VocabStruct
from vectorizing.labse_embedder import SentenceEmbedder
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import pandas as pd
import json
from helpers.help_functions import just_non_zero_values_dict
from topic import TopicURL
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sentence_transformers import util
from tqdm import tqdm
import numpy as np
from helpers.text_cleaning import split_sentences

from keybert import KeyBERT


# for test
from test_data import test_dictionary
from test_data import keywords
from test_data import descriptions
from test_data import titles
from test_data import contents


print('start')

stopwords = ['Главная', 'Контакты', 'Доставка', 'Оплата', 'МОБИЛЬНАЯ ВЕРСИЯ']
stopwords = [word.lower() for word in stopwords]

sentence_embedder = SentenceEmbedder(device='cuda', path_enc_configs='LaBSE')
agg = AgglomerativeClustering(affinity='cosine', linkage='average') 
clf = NearestCentroid(metric='cosine')
kw_model = KeyBERT('LaBSE')



for i, url in tqdm(enumerate(test_dictionary)):
    print()
    print('===='*30)
    print()
    print(f'URL : {url}')
    #test data
    curr_keywords = keywords[i]
    print(f'keywords: {curr_keywords}')
    
    for range_iter in range(2, 3):
        if bool(descriptions):
            curr_description = list() #sent_tokenize(description[i].strip())
            curr_description.extend(
                list(dict(kw_model.extract_keywords(descriptions[i], use_mmr=True, diversity=0.8, keyphrase_ngram_range=(1, range_iter), stop_words=None)).keys()))
            print(f'description : {curr_description}')
        if bool(titles):    
            curr_title = list() #sent_tokenize(title[i].strip())
            curr_title.extend(
                list(dict(kw_model.extract_keywords(titles[i], use_mmr=True, diversity=0.8,  keyphrase_ngram_range=(1, range_iter), stop_words=None)).keys()))
            print(f'title: {curr_title}')
        # if bool(content):
        #     curr_content = word_tokenize(content[i])
        if bool(contents):
            # curr_content = ' '.join(sent_tokenize(' '.join(sent_tokenize(content[i])))).split('  ')
            # curr_content = [token for token in curr_content if token.lower() not in stopwords]
            # curr_content = word_tokenize(content[i])
            curr_content = list()
            curr_content.extend(
                list(dict(
                    kw_model.extract_keywords(
                        contents[i], use_mmr=True, diversity=0.8,  keyphrase_ngram_range=(1, range_iter), stop_words=None)).keys()
                    )
                )
            print(f'Content : {curr_content}')

    url_struct = URLStructure(
        url_path=url,
        keywords=curr_keywords,
        description=curr_description,
        title=curr_title,
        content=curr_content,
        fill_dict=True,
        print_feedback_tokens=True
    )

    url_struct.fill_embed_fields(embedder=sentence_embedder)
    url_struct.form_labels_centroid_maintokens(
        agg_clusterer=agg, centroid_clf=clf)
    url_struct.form_output_embeddings(
        agg_clusterer=agg)
            
        
        


print('finish with OK')