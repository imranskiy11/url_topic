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
from tqdm import tqdm


# for test
from test_data import test_dictionary
from test_data import keywords as kw
from test_data import descriptions as desc
from test_data import titles as ttls
from test_data import contents as cntnt


def main():
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
    
    topic = TopicURL(vocab_embeddings=vocab_struct.embedded_vocab)
    
    print(colored(f'Running iterations', 'green'))
    start_time = time.time()
    for i, url in tqdm(enumerate(test_dictionary)):
        print(f'URL : {url}')
        #test data
        keywords = kw[i]
        description = sent_tokenize(desc[i])
        title = sent_tokenize(ttls[i])
        content = sent_tokenize(cntnt[i])
    
    
        url_struct = URLStructure(
            url_path=url,
            keywords=keywords,
            description=description,
            title=title,
            content=None,
            fill_dict=True
        )
        
        url_struct.fill_embed_fields(embedder=sentence_embedder)
        url_struct.form_labels_centroid_maintokens(
            agg_clusterer=agg, centroid_clf=clf, verbose=verbose, save_feedback_tokens=save_feedback_tokens)
        url_struct.form_output_embeddings(
            agg_clusterer=agg, verbose=verbose, save_feedback_tokens=save_feedback_tokens)
        
        output_embeddings = url_struct.output_summary_embeddings
        
        distances_vocab = just_non_zero_values_dict(
            topic.run(output_embeddings, b_value=0.35, threshold_other=0.3))
        predicted_list.append(list(distances_vocab.keys()))
        
        if len(set(list(distances_vocab.keys())[:3]).intersection(set(test_dictionary[url]))) > 0:
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
    main()
