"""

Jonathan
"""
import json, argparse, os

import pandas as pd
import numpy as np

from datetime import datetime as dt 

def get_data(filename = 'MAGtest.json'):
    with open('../original-data/MAG/{0}'.format(filename), "rb") as json_file:
        raw_data = json.load(json_file)

    data = pd.DataFrame(raw_data)

    assert len(raw_data) == len(data.index)

    return data

def _fix_image_file(filename = 'MAGimageID.csv'):
    """
    In this function, we clean the imageID file so that comma's, etc.,
    are removed from the filenames.
    """
    import csv

    with open('../data/{0}'.format(filename), 'r') as infile, open('../data/fixed_ids.csv', 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter='|')
        for line in reader:
            newline = [''.join(line)]
            writer.writerow(newline)

    return pd.read_csv('../data/fixed_ids.csv', delimiter='|')

def get_image_ids(filename = 'MAGimageID.csv'):
    image_data = pd.read_csv('../data/{0}'.format(filename), usecols=['Identifier'])

    image_data.columns = map(str.lower, image_data.columns)

    image_data['identifier'] = image_data['identifier'].apply(lambda x: x.replace(' ', '').lower())

    return image_data

def check_image_ids(data, images):
    if len(images) == 0: return 0 # can't do anything in this case

    images_set = set(images)

    # follow the format {<row_index>: [<missing_file_1>, ...]} so we know what's missing.
    missing_images = {}
    for row in data.itertuples(): # for each row in the dataframe
        for value in row.identifier_formatted: # for each identifier in the list of identifiers
            if value not in images_set: # if that value is not in the image set
                missing_images.setdefault(row.Index, []).append(value) # add it to the dictionary

    return missing_images


def process_data(data):
    """
    This is the bulk of the data processing. Here we clean and add
    additional, useful columns to the data.
    """
    if len(data) == 0: return 0

    # set all of the column names to lowercase (it's just better...)
    data.columns = map(str.lower, data.columns)

    # process the file identifiers, i.e. the image names. We'll turn them into a list
    data['identifier'] = data['identifier'].apply(lambda x: x.splitlines())

    # add in a feature that describes how many files are assoicated with the metadata row
    data['number_of_files'] = data['identifier'].apply(lambda x: len(x))

    # drop the rows where number_of_files is zero
    data = data[data.number_of_files != 0]

    # add in a feature that describes how many characters are in the description (useful for filtering)
    data['len_description'] = data['description (physical 1)'].apply(lambda x: len(x))

    # we need a column that turns the image identifiers into the same format as the image set (no whitespace + lowercase)
    data['identifier_formatted'] = data['identifier'].apply(lambda x: [y.replace(' ', '').lower() for y in x])

    # remove new line characters in the following set of columns
    cols = [
        'creator\'s name', 'date of birth', 'date of death',
        'technique: (technique and material)', 'medium: (technique and material)',
        'description (physical 1)', 'material: (technique and material)'
    ]

    for col in cols:
        data[col] = data[col].replace(r'\n', ';', regex = True)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image metadata file')
    parser.add_argument('--file', action='store', help='Which metadata file to process')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output file')
    args = parser.parse_args() # parse those args.

    filename = 'MAGtest.json'
    if args.file:
        filename = args.file
    
    start_ts = dt.now()

    print('[{0}] Reading in data file: {1}...'.format(dt.now(), filename))
    data = get_data(filename = filename)

    # process the data
    print('[{0}] Process the data...'.format(dt.now()))
    data = process_data(data)

    if not os.path.exists('../data/fixed_ids.csv'):
        _fix_image_file()

    # this processes the image identifiers so there in a common format: no whitespace and lowercase.
    image_identifiers = get_image_ids(filename = 'fixed_ids.csv')

    # check that the we have the images
    missing_images = check_image_ids(data, image_identifiers['identifier'].values.tolist())

    number_of_files = sum(data['number_of_files'])
    number_of_missing_files = sum([len(v) for k, v in missing_images.items()])
    print(number_of_files, number_of_missing_files, number_of_files - number_of_missing_files)

    if os.path.exists('../data/{0}_processed.csv'.format(filename.split('.')[0])) and not args.overwrite:
        print('[{0}] Processed file ({1}_processed.csv) already exists, please provide the --overwrite flag...'.format(
            dt.now(), filename.split('.')[0]))
    else:
        data.to_csv('../data/{0}_processed.csv'.format(filename.split('.')[0]), index=False, sep='|')

    end_ts = dt.now()
    print('[{0}] Finished processing data... total time: {1}'.format(end_ts, end_ts - start_ts))
