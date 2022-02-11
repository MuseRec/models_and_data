"""
"""
import pickle as p 
import numpy as np 
import matplotlib.pyplot as plt 
import faiss as f_ai
import json as j

from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.random.seed(42)


def metadata_model():
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
    index = f_ai.IndexFlatIP(25)
    index.add(vectors)
    
    # write the 'model' to file
    f_ai.write_index(index, 'models/faiss_metadata_only')

    # write the padded vectors to file - for consistent use when concat'd with images
    for idx, (f_name, _) in enumerate(metadata_vectors.copy().items()):
        metadata_vectors[f_name] = vectors[idx]

    p.dump(metadata_vectors, open('data/metadata/metadata_vectors_padded.pickle', 'wb'))

def encoded_images_model():
    image_vectors = p.load(open('data/images/encoded_imgs.pickle', 'rb'))

    # get the vectors without the file names, we'll match them back up later
    vectors = [vec for _, vec in image_vectors.items()]

    # normalise the vectors prior to indexing so the FlatIP index is the same as cosine similarity
    vectors = normalize(vectors, norm = 'l2')

    # the normalize step changes the dtype to float64, but it needs to be float32 for faiss
    vectors = np.asarray([vec.astype(np.float32) for vec in vectors])

    # build the index and add the data
    index = f_ai.IndexFlatIP(512)
    index.add(vectors)

    # write the model to file
    f_ai.write_index(index, 'models/faiss_encoded_images')

    # write the normalised vectors to file
    for idx, (f_name, _) in enumerate(image_vectors.copy().items()):
        image_vectors[f_name] = vectors[idx]

    p.dump(image_vectors, open('data/images/encoded_imgs_normalised.pickle', 'wb'))

    # write the filename order to disk - will probably be useful
    j.dump(
        [f_name for f_name, _ in image_vectors.items()],
        open('data/images/encoded_images_order.json', 'w')
    )

    return image_vectors


def concatenated_model():
    # get the metadata vectors
    metadata_vectors = p.load(open('data/metadata/metadata_vectors_padded.pickle', 'rb'))

    # get the image vectors
    image_vectors = p.load(open('data/images/encoded_imgs_normalised.pickle', 'rb'))

    # concatenate the vectors
    concatenated_vectors = {}
    for f_name, vector in metadata_vectors.items():
        concatenated_vectors[f_name] = np.concatenate((image_vectors[f_name], vector))
        
    # build the index and add the data (537 is the concatenated length)
    index = f_ai.IndexFlatIP(537)
    index.add(np.array([vec for _, vec in concatenated_vectors.items()]))

    # write the model to file 
    f_ai.write_index(index, 'models/faiss_concatenated')

    # write the vectors to file
    p.dump(concatenated_vectors, open('data/images/concatenated_vectors.pickle', 'wb'))


def main():
    # metadata_model()
    # encoded_images_model()
    concatenated_model()
       

if __name__ == '__main__':
    main()