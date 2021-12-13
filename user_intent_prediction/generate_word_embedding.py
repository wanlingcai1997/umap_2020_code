from gensim.models import word2vec
import numpy as np
import sys
import helper

def extract_word_embedding_from_glove(embedding_dimension):
    word_embeddings_index = dict()
    file_GloVe = open('../../additional_resource/glove.6B/glove.6B.%dd.txt' % embedding_dimension, encoding='utf-8')
    for line in file_GloVe:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings_index[word] = coefs
    file_GloVe.close()
    print('Loaded %s word vectors.' % len(word_embeddings_index))
    return word_embeddings_index

def create_word_embedding_matrix(embedding_dimension, num_of_words, word_index):
    word_embeddings_index = extract_word_embedding_from_glove(embedding_dimension)
    embedding_matrix = np.zeros((num_of_words, embedding_dimension))
    for word, index in word_index.items():
        if index >= num_of_words:
            continue
        embedding_vector = word_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector

    return embedding_matrix
    