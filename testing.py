
from subprocess import CREATE_DEFAULT_ERROR_MODE
import warnings
warnings.filterwarnings('ignore')

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
from test_data import keywords as kw
from test_data import descriptions as desc
from test_data import titles as ttls
from test_data import contents as cntnt




     


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
        scores.append(np.mean(np.array(means)))
    return np.around(np.array(scores), 3)


def topic_run(keywords=None, description=None, title=None, content=None, print_feedback_tokens=True):
    print(colored(f'Starting /// load embedder and models', 'green'))
    sentence_embedder = SentenceEmbedder()
    agg = AgglomerativeClustering(affinity='cosine', linkage='average') 
    clf = NearestCentroid(metric='cosine')

    ########## for metrics ##########
    marks = list()
    predicted_list = list()
    result_table_df = pd.DataFrame()
    
    save_feedback_tokens = False
    verbose = 0
    #################################
    
    print(colored(f'Loading vocab classes data', 'green'))
    with open('topic_vocab/vocab.json', 'r') as json_file:
        classes_vocab = json.load(json_file)
    
    print(colored(f'Building vocab classes data structure', 'green'))   
    vocab_struct = VocabStruct(classes_vocab, metric='cosine')
    vocab_struct.fill_embed_vocab(embedder=sentence_embedder)
    
    vocab_struct.fill_centroid_vocab()
    

    
    stopwords = ['Главная', 'Контакты', 'Доставка', 'Оплата', 'МОБИЛЬНАЯ ВЕРСИЯ']
    stopwords = [word.lower() for word in stopwords]
    
    topic = TopicURL(vocab_embeddings=vocab_struct.embedded_vocab)
    
    kw_model = KeyBERT('LaBSE')
    
    print(colored(f'Running iterations', 'green'))
    
    start_time = time.time()
    for i, url in tqdm(enumerate(test_dictionary)):
        print(f'URL : {url}')
        #test data
        curr_keywords = keywords[i]
        print(f'keywords: {curr_keywords}')
        
        range_iter = 3
        if bool(description):
            curr_description = sent_tokenize(description[i].strip())
            curr_description = list() #
            curr_description.extend(
                    list(
                        {k:v for k, v in kw_model.extract_keywords(
                        description[i],
                        use_maxsum=True,  
                        use_mmr=False, 
                        # diversity=0.8,  
                        keyphrase_ngram_range=(1, range_iter), 
                        stop_words=stopwords, 
                        nr_candidates=20, 
                        top_n=5
                    ) if v < 0.7}.keys()
                    )
                )
            print(f'description : {curr_description}')
        if bool(title):    
            # curr_title = sent_tokenize(title[i].strip())
            curr_title = list() #
            curr_title.extend(
                    list(
                        {k:v for k, v in kw_model.extract_keywords(
                        title[i],
                        use_maxsum=True,  
                        use_mmr=False, 
                        # diversity=0.8,  
                        keyphrase_ngram_range=(1, range_iter), 
                        stop_words=stopwords, 
                        nr_candidates=20, top_n=5
                    ) if v < 0.7}.keys()
                    )
                )
            print(f'title: {curr_title}')
            # if bool(content):
            #     curr_content = word_tokenize(content[i])
        if bool(content):
            pass
        #     # curr_content = ' '.join(sent_tokenize(' '.join(sent_tokenize(content[i])))).split('  ')
        #     # curr_content = [token for token in curr_content if token.lower() not in stopwords]
        #     # curr_content = word_tokenize(content[i])
            curr_content = list()
            curr_content.extend(
                list(
                    {k:v for k, v in kw_model.extract_keywords(
                    content[i],
                    use_maxsum=True,  
                    use_mmr=False, 
                    # diversity=0.8,  
                    keyphrase_ngram_range=(1, range_iter), 
                    stop_words=stopwords, 
                    nr_candidates=20, top_n=5
                ) if v < 0.7}.keys()
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
            print_feedback_tokens=print_feedback_tokens
        )
        
        url_struct.fill_embed_fields(embedder=sentence_embedder)
        url_struct.form_labels_centroid_maintokens(
            agg_clusterer=agg, centroid_clf=clf)
        url_struct.form_output_embeddings(
            agg_clusterer=agg)
        
        output_embeddings = url_struct.output_summary_embeddings
        
        ##################
        # print(f'output embs shape: {output_embeddings.shape}')
        
        
        # scores_dist_dict = dict()
        # for vocab_cls in vocab_struct.embedded_vocab.keys():
        #     # print(f'{vocab_cls}: {vocab_struct.embedded_vocab[vocab_cls].shape}')
        #     dists = util.semantic_search(vocab_struct.embedded_vocab[vocab_cls], output_embeddings)
        #     scores = get_scores_from_semantic_search(dists)
            
        #     scores_median = np.around(np.median(scores), 2)
        #     scores_mean = np.around(np.mean(scores), 2)
        #     scores_max = np.around(np.max(scores), 2)
        #     scores_min = np.around(np.min(scores), 2)
        #     score_cls = np.around((scores_median + scores_mean)/2, 2)

        #     scores_dist_dict[vocab_cls] = (scores_mean + scores_median)/2

        ##############
            
        
        if print_feedback_tokens:
            print(f'Feedback tokens : {url_struct.output_feedback_tokens}')
            
        distances_vocab = just_non_zero_values_dict(
            topic.run(output_embeddings, b_value=0.35, threshold_other=0.3))
        predicted_list.append(list(distances_vocab.keys()))
        pred_topic = list(distances_vocab.keys())
        
        # pred_topic = max(scores_dist_dict, key=scores_dist_dict.get)
        # distances_vocab = {k:v for k,v in scores_dist_dict.items() if k == pred_topic}
        # predicted_list.append(list(distances_vocab.keys()))
        print(f'distance vocab: {distances_vocab} ||| {test_dictionary[url]}')
        
        
        
        
        # print(f'Predicted topics : {pred_topic}\n\n')
        
        if len(set(list(distances_vocab.keys())[:2]).intersection(set(test_dictionary[url]))) > 0:
            marks.append(1)
        else:
            marks.append(0)

        print('\n\n')
    spended_time = round(time.time() - start_time, 2)
    print(colored(f'Finish ///', 'green'), colored(f'spended time : {spended_time}', 'red'))  
          
    marks = pd.Series(marks)
    result_table_df['URL'] = (test_dictionary.keys())
    result_table_df['True'] = list(test_dictionary.values())
    result_table_df['Predict'] = predicted_list
    result_table_df['Predict Size'] = result_table_df.Predict.apply(len)
    # result_table_df.to_csv('result.csv', index_label=False)
    
        
    print(result_table_df)
    
    print(f'result : {len(test_dictionary)} : {marks[marks == 1].shape} || {marks[marks == 0].shape}')
    print(f'accuracy : {round(marks[marks == 1].shape[0]*100/len(test_dictionary), 2)} %')
    print(colored(f'Average time on website : {round(spended_time/len(test_dictionary), 2)}', 'cyan'))


    


if __name__ == '__main__':
    topic_run(kw, desc, ttls, cntnt)