"""

"""

import pickle as p 
import numpy as np
import pandas as pd   

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

def metadata_similarity():
    metadata_vectors = p.load(open('data/metadata/metadata_vectors.pickle', 'rb'))

    # create an index mapping for later
    index_mapping = {idx: img for idx, (img, _) in enumerate(metadata_vectors.items())}

    # for padding, get the vectors without the filenames, we'll match them back up
    vectors = [vec for _, vec in metadata_vectors.items()]

    # we need to pad the vectors to a consistent length.
    vectors = pad_sequences(
        vectors, maxlen = 25,
        dtype = 'float32', padding = 'post',
        truncating = 'post', value = 0
    )
    
    # calculate the cosine similarity (normalisation is done in this process)
    similarities = cosine_similarity(vectors)

    # get the sorted indexes in reverse order
    sorted_index = np.argsort(-similarities)

    # the results want to be in the form of:
    # [(img, sim_img, score, data_rep), (...)]
    similarity_results = []

    for idx, sorted_idx in enumerate(sorted_index):
        # take the top-10
        highest_ranked = sorted_idx[1:11]

        for ranked in highest_ranked:
            similarity_results.append((
                index_mapping[idx], # the image 
                index_mapping[ranked], # the similar image 
                round(similarities[idx][ranked], 4), # the score
                'meta' # record that it's the meta vectors
            ))

    # save to file
    df = pd.DataFrame(
        similarity_results, columns = ['img', 'sim_img', 'score', 'data_rep']
    )
    print(df.head(20))
    df.to_csv('data/metadata/meta_similarties.csv', index = False)
    # j.dump(similarity_results, open('data/metadata/meta_similarties.json', 'w'))

    # save the padded vectors
    for idx, (f_name, _) in enumerate(metadata_vectors.copy().items()):
        metadata_vectors[f_name] = vectors[idx]

    p.dump(metadata_vectors, open('data/metadata/metadata_vectors_padded_new.pickle', 'wb'))


def image_similarity():
    image_vectors = p.load(open('data/images/encoded_imgs.pickle', 'rb'))

    # create an index mapping for later
    index_mapping = {idx: img for idx, (img, _) in enumerate(image_vectors.items())}

    # get the vectors for the similarity calculation
    vectors = [vec for _, vec in image_vectors.items()]

    # calculate the cosine similarity
    similarities = cosine_similarity(vectors)

    # get the sorted indexes in reverse order
    sorted_index = np.argsort(-similarities)

    # the results want to be in the form of:
    # [(img, sim_img, score, data_rep), (...)]
    similarity_results = []

    for idx, sorted_idx in enumerate(sorted_index):
        # take the top-10
        highest_ranked = sorted_idx[1:11]

        for ranked in highest_ranked:
            similarity_results.append((
                index_mapping[idx], # the image 
                index_mapping[ranked], # the similar image 
                round(similarities[idx][ranked], 4), # the score 
                'image'
            ))

    # save to file 
    df = pd.DataFrame(
        similarity_results, columns = ['img', 'sim_img', 'score', 'data_rep']
    )
    print(df.head(20))
    df.to_csv('data/images/image_similarities.csv', index = False)

def concatenated_similarity():
    # get the metadata vectors
    metadata_vectors = p.load(open('data/metadata/metadata_vectors_padded_new.pickle', 'rb'))

    # get the image vectors
    image_vectors = p.load(open('data/images/encoded_imgs.pickle', 'rb'))

    # create an index mapping for later
    index_mapping = {idx: img for idx, (img, _) in enumerate(metadata_vectors.items())}

    # concatenate the vectors
    concatenated_vectors = {}
    for f_name, vector in metadata_vectors.items():
        concatenated_vectors[f_name] = np.concatenate((image_vectors[f_name], vector), axis = 0)
    
    # get the vectors for the similarity calculation
    vectors = [vec for _, vec in concatenated_vectors.items()]

    # calculate the similarity 
    similarities = cosine_similarity(vectors)

    # get the sorted indexes in reverse order
    sorted_index = np.argsort(-similarities)

    # the results want to be in the form of:
    # [(img, sim_img, score, data_rep), (...)]
    similarity_results = []

    for idx, sorted_idx in enumerate(sorted_index):
        # take the top-10
        highest_ranked = sorted_idx[1:11]

        for ranked in highest_ranked:
            similarity_results.append((
                index_mapping[idx], # the image
                index_mapping[ranked], # the similar image 
                round(similarities[idx][ranked], 4), # the score
                'concatenated'
            ))
    
    # save to file
    df = pd.DataFrame(
        similarity_results, columns = ['img', 'sim_img', 'score', 'data_rep']
    )
    print(df.head(20))
    df.to_csv('data/images/concatenated_similarities.csv', index = False)


def main():
    # metadata_similarity()
    # image_similarity()
    concatenated_similarity()

if __name__ == '__main__':
    main()