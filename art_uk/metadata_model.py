"""

"""

import pandas as pd 
import numpy as np 
import itertools 
import string 
import json 
import os
import re
import traceback 

class Metadata:

    def __init__(self, metadata):
        self.metadata = metadata
        self.process_data()

    def __iter__(self):
        for _, meta in self.data:
            yield meta

    def process_data(self):

        def _linked_topics(topics):
            if not topics:
                return topics 

            topics_arr = []
            for topic in topics:
                if ' and ' in topic:
                    sub_topic_str = topic.strip().replace(' and ', ' ').split(' ')
                    for sub_str in sub_topic_str:
                        topics_arr.append(sub_str.strip())
                else:
                    topics_arr.append(topic.strip())

            # join the topics into a single string for downstream usefulness
            return ' '.join(topics_arr)

        def _linked_terms(terms):
            if not terms:
                return terms
            
            terms_arr = [
                term.strip() for term in terms 
            ]

            return ' '.join(terms_arr)

        def _artwork_title(titles):
            if not titles:
                return titles

            #print(titles)
            
            removed_dates = ''.join([i for i in titles if not i.isdigit()])
            print(removed_dates)
                 
            removed_punctuation = removed_dates.translate(str.maketrans('', '', string.punctuation))
            print(removed_punctuation)
            
            #return removed_punctuation

            removed_hyphens = removed_punctuation.replace('–', '')

            print(removed_hyphens)
            return removed_hyphens
        
        self.metadata[['Linked Terms', 'Linked Topics']] = self.metadata[[
            'Linked Terms', 'Linked Topics']].replace(np.nan, 0)
        
        # Terms
        self.metadata['Linked Terms'] = self.metadata['Linked Terms'].apply(
            lambda x: x.split(',') if x is not 0 else ''
        )
        self.metadata['Linked Terms'] = self.metadata['Linked Terms'].apply(_linked_terms)

        # Topics
        self.metadata['Linked Topics'] = self.metadata['Linked Topics'].apply(
            lambda x: x.split(',') if x is not 0 else ''
        )
        self.metadata['Linked Topics'] = self.metadata['Linked Topics'].apply(_linked_topics)

        #remove the punctation from the title and then strip any leading whitespace
        # self.metadata['Artwork Title'] = self.metadata['Artwork Title'].apply(
        #     lambda x: x.translate(str.maketrans('', '', string.punctuation)).split()
        # )
        self.metadata['Artwork Title'] = self.metadata['Artwork Title'].apply(_artwork_title)
        
        # create dataset used in the iterator
        self.data = {
            file_name: ' '.join(meta).strip()
            for file_name, meta in zip(
                self.metadata['Filename'].values.tolist(), 
                self.metadata[['Artwork Title', 'Linked Terms', 'Linked Topics']].values.tolist()
            )
        }
        
        # try:
        #     self.data = {}
        #     current_meta = None 
        #     current_filename = None 
        #     for file_name, meta in zip(
        #         self.metadata['Filename'].values.tolist(), 
        #         self.metadata[['Artwork Title', 'Linked Terms', 'Linked Topics']].values.tolist()
        #     ):
        #         current_meta = meta 
        #         current_filename = file_name
        #         self.data[file_name] = ' '.join(meta).strip()
        # except TypeError:
        #     print(current_meta)
        #     print(current_filename)
        #     print(traceback.format_exc())
          
def main():
    md = Metadata(
        metadata = pd.read_csv('../data/Art-UK/ArtUK_main_sample.csv', sep = '|')
    )

    #print(md.data['LAN_PSGA_PSGCT_2018_1-001.jpg'], type(md.data['LAN_PSGA_PSGCT_2018_1-001.jpg']))

    print(string.punctuation)
    # exit()

    # save the strings produced by the metadata class
    if not os.path.isfile('../data/Art-UK/metadata_strings.json'):
        json.dump(md.data, open('../data/Art-UK/metadata_strings.json', 'w'), indent = 4)

    # pass md to word2vec

    # get trained vectors from word2vec (skip-gram)

    # match with metadata ids

    # save as pickle object


if __name__ == '__main__':
    main()