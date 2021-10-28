from typing import cast
import pandas as pd
import os
import glob
from pandas.errors import ParserError
import numpy as np

np.random.seed(42)

# use glob to get all the csv files in the path
def get_data():
    path = '../data/Art-UK'
    csv_files = glob.glob(path + "/*.csv")

    #Create a concatenated file from all ArtUK csv files
    df = pd.concat((pd.read_csv(f) for f in csv_files))
    print(len(df))
    return df

def get_image_ids():
    image_list = []

    for f in glob.glob('/Users/user/Desktop/ImageExportLukasN/**/*.jpg', recursive = True):
        f_name = f.split('/')[-1]
        image_list.append(f_name)
    print(len(image_list))
    print(image_list[0:10])
    return image_list

#match the image IDs to the IDs stored in the csv
def meta_data_match(df, image_list):
    metadataFilenames = df['Filename'].values.tolist()
    matched_ids = set(metadataFilenames).intersection(set(image_list))
    
    print(len(matched_ids))
    return matched_ids 

def concat_file(df, matched_ids):
    df = df[df['Filename'].isin(matched_ids)].copy()
    print(len(df))
    print(df.head(2))
    return df

def drop_columns(df):
    drop_df = df.drop(columns=['ART UK ARTWORK ID', 'Collection Mnemonic', 'Location', 'Title Addtional Information', 'Collection Accession Number', 'Art UK accession number', 'No Image', 'Artwork Dbpedia ID', 'Artist Role',
    'Artist ID\'s', 'Uncertain Attributions', 'Linked Region Outdoor Sculpture', 'Acquisition Method', 'Private Loan', 'Permanent URL Link to Collection', 'Permanent URL Link to Print on Demand Service', 'Notes', 'Art UK-Location Notes', 'Owner',
    'Custodian', 'Installation Start Date', 'Installation End Date', 'Unveiling Date', 'Listing date', 'Sculpture sited outside', 'Signature/marks description', 'Date of Photograpohy/Recording', 'Image Views Taken', 'Street', 'Town/City', 'Postcode', 'Latitude', 'Longitude',
    'OS grid reference', 'Setting Note', 'Access Note', 'Artwork Heights', 'Artwork Widths', 'Artwork Depths', 'Artwork Estimates', 'Plinth Heights', 'Plinth Widths', 'Plinth Depths', 'Plinth Estimates', 'Relative Path', 'Date Created', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66'])
    return drop_df

#generating a 30% sample out of all entries
def master_sampled(drop_df):
    ArtUK_main_sample = drop_df.sample(frac =.15, random_state = 42)
    print(len(ArtUK_main_sample))
    return ArtUK_main_sample








def main():
    df = get_data()

    image_list = get_image_ids()

    matched_ids = meta_data_match(df, image_list)

    df = concat_file(df, matched_ids)

    drop_df = drop_columns(df)
    
    ArtUK_main_sample = master_sampled(drop_df)
    ArtUK_main_sample.to_csv('ArtUK_main_sample.csv', sep='|', index = False)





    



if __name__ == '__main__':
    main()
