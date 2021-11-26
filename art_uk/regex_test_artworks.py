import pandas as pd 
import numpy as np 
import itertools
import re
import string

col_list = ['Artwork Title', 'Filename']


class Metadata:

    def __init__(self, metadata):
        self.metadata = metadata
        self.process_data()

    def __iter__(self):
        for _, meta in self.data:
            yield meta

    def process_data(self):

        def _artwork_titles(titles):
            if not titles:
                return titles

            #print(titles)
            
            
            
            #removed_dates =  re.sub(pattern, '', titles)
            removed_dates = ''.join([i for i in titles if not i.isdigit()])
            print(removed_dates)
                 
            removed_punctuation = removed_dates.translate(str.maketrans('', '', string.punctuation))
            print(removed_punctuation)
            
            #return removed_punctuation

            removed_hyphens = removed_punctuation.replace('–', '')

            print(removed_hyphens)
            return removed_hyphens

        self.metadata['Artwork Title'] = self.metadata['Artwork Title'].apply(_artwork_titles)


        self.data = {
            file_name: ' '.join(meta).strip()
            for file_name, meta in zip(
                self.metadata['Filename'].values.tolist(), 
                self.metadata[['Artwork Title']].values.tolist()
            )
        }


def main():
    md = Metadata(
        metadata = pd.read_csv('../data/Art-UK/ArtUK_main_sample.csv', usecols=col_list, sep = '|')
    )

    

if __name__ == '__main__':
    main()