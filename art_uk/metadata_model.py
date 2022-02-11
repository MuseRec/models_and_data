"""
Preprocessing of the ArtUK metadata and gensim model to produce metadata vectors
"""

import pandas as pd 
import numpy as np 
import string 
import json 
import os
import gensim.downloader
import pickle

from collections import OrderedDict


class Metadata:

    def __init__(self, metadata):
        self.metadata = metadata
        self.process_data()

    def __iter__(self):
        for _, meta in self.data:
            yield meta

    def process_data(self):

        def _linked_topics(topics):
            if isinstance(topics, str) and topics.isspace():
                return " " 

            topics_arr = []
            for topic in topics:
                if ' and ' in topic:
                    sub_topic_str = topic.strip().replace(' and ', ' ').split(' ')
                    for sub_str in sub_topic_str:
                        topics_arr.append(sub_str.strip())
                else:
                    topics_arr.append(topic.strip())

            # join the topics into a single string for downstream usefulness
            # hacky method to get rid of the whitespace
            joined_topics = " ".join(topics_arr)

            # remove punctuation
            joined_topics = joined_topics.translate(str.maketrans('', '', string.punctuation))

            return " ".join(joined_topics.split())

        def _linked_terms(terms):
            if isinstance(terms, str) and terms.isspace():
                return " "
            
            terms_arr = " ".join([
                term.strip() for term in terms 
            ])

            # remove punctuation
            terms_arr = terms_arr.translate(str.maketrans('', '', string.punctuation))

            # hacky method to get rid of the whitespace
            return " ".join(terms_arr.split())

        def _artwork_title(titles):
            if not titles:
                return " "

            # remove date from title
            removed_dates = ''.join([i for i in titles if not i.isdigit()])

            # remove the punctuation in titles  
            removed_punctuation = removed_dates.translate(str.maketrans('', '', string.punctuation))
        
            #return removed_punctuation
            removed_hyphens = removed_punctuation.replace('–', '')

            return " ".join(removed_hyphens.split())
        
        self.metadata[['Linked Terms', 'Linked Topics']] = self.metadata[[
            'Linked Terms', 'Linked Topics']].replace(np.nan, 0)
        
        # Terms
        self.metadata['Linked Terms'] = self.metadata['Linked Terms'].apply(
            lambda x: x.split(',') if x is not 0 else ' '
        )
        self.metadata['Linked Terms'] = self.metadata['Linked Terms'].apply(_linked_terms)

        # Topics
        self.metadata['Linked Topics'] = self.metadata['Linked Topics'].apply(
            lambda x: x.split(',') if x is not 0 else ' '
        )
        self.metadata['Linked Topics'] = self.metadata['Linked Topics'].apply(_linked_topics)

        self.metadata['Artwork Title'] = self.metadata['Artwork Title'].apply(_artwork_title)
        
        # create dataset used in the iterator
        self.data = {
            file_name: " ".join(' '.join(meta).strip().split()).lower()
            for file_name, meta in zip(
                self.metadata['Filename'].values.tolist(), 
                self.metadata[['Artwork Title', 'Linked Terms', 'Linked Topics']].values.tolist()
            )
        }

          
def main():
    # ../data/Art-UK/ArtUK_main_sample.csv
    md = Metadata(
        metadata = pd.read_csv('data/metadata/artuk-metadata-subset.csv', sep = '|')
    )

    # save the strings produced by the metadata class
    if not os.path.isfile('data/metadata/metadata_strings.json'):
        json.dump(md.data, open('data/metadata/metadata_strings.json', 'w'), indent = 4)
    
    #load pre-trained model (glove-wiki-50)
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

    #create a dict to store avergaed vectors in
    # metadata_vectors = {}
    metadata_vectors = OrderedDict()

    #iterate over all words (aka values) per key in metadata file
    for filename, words in md.data.items():
        artuk_vectors = []
        
        for word in words.split(" "): #split by word
            try: 
                artuk_vectors.append(glove_vectors.get_vector(word)) #create vector for word
            except KeyError:
                #artuk_vectors.append(np.array([]))
                continue
                
        try:
            artuk_vectors_avg = np.mean(artuk_vectors, axis=1) #calculate word vector avergage per row
        except np.AxisError:
            artuk_vectors_avg = []
         
        f_name = filename.split(os.sep)[-1].split('.jpg')[0]
        metadata_vectors[f_name] = artuk_vectors_avg #save every averaged vector (value) to filename (key)

    #little test to check for empty arrays
    empty_count, non_empty_count = 0, 0
    len_arr = []
    for f_name, vectors in metadata_vectors.items():
        len_arr.append(len(vectors))
        if len(vectors) == 0:
            empty_count += 1
        else:
            non_empty_count += 1

    print(empty_count, non_empty_count) 
    print(np.mean(len_arr), np.std(len_arr))

    #create pickle object to store averaged vectors
    with open('data/metadata/metadata_vectors.pickle', 'wb') as handle:
        pickle.dump(metadata_vectors, handle)

if __name__ == '__main__':
    main()