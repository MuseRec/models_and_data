"""
"""
import pickle as p 
import numpy as np 
import matplotlib.pyplot as plt 
import faiss

from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    metadata_vectors = p.load(open('data/metadata/metadata_vectors.pickle', 'rb'))

    # get the vectors without the file names, we'll match them back up later
    vectors = [list(vec) for _, vec in metadata_vectors.items()]

    # we need to pad the vectors to a consist length, both normalization and faiss rely on it
    # we pad after the vector and up to a length of 25 - the majority of vectors are less than 20
    vectors = pad_sequences(
        vectors, maxlen = 25, 
        dtype = 'float32', padding = 'post',
        truncating = 'post', value = 0.
    )

    # normalise prior to indexing so the FlatIP index is the same as cosine simularity 
    vectors = normalize(vectors, norm = 'l2')

    # build the index and add the data
    index = faiss.IndexFlatIP(25)
    index.add(vectors)
    
    # write the 'model' to file
    faiss.write_index(index, 'models/faiss_metadata_only')

    # write the padded vectors to file - for consistent use when concat'd with images
    for idx, (f_name, _) in enumerate(metadata_vectors.copy().items()):
        metadata_vectors[f_name] = vectors[idx]

    p.dump(metadata_vectors, open('data/metadata/metadata_vectors_padded.pickle', 'wb'))
       

if __name__ == '__main__':
    main()