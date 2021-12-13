# Train.py
import numpy as np
import pandas as pandas
from Data import Data

# Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

# from user intent prediction code
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional, LSTM
from keras.models import Model

class BuildModel():
    
    @staticmethod
    def cnn_base_model():

        # -----------------------------------------------------
        # Embedding Layer (input)
        # -----------------------------------------------------

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        if Data.embedding_selection == 'glove' or Data.embedding_selection == 'word2vec' :
            
            embedding_layer = Embedding(Data.actual_num_of_vocabulary,
                                    Data.embedding_dimension,
                                    weights=[Data.embedding_matrix],
                                    input_length=Data.final_max_seq_len,
                                    trainable=False)

        else:
            embedding_layer = Embedding(Data.actual_num_of_vocabulary,
                                    Data.embedding_dimension,
                                    embeddings_initializer='uniform', 
                                    input_length=Data.final_max_seq_len)
        
        # -----------------------------------------------------
        # CNN - Convolutional Layer
        #     - Max Pooling Layer
        #     - Dropout Layer 
        #     - Dense Layer
        # -----------------------------------------------------


        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(Data.final_max_seq_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(Data.convolutional_units, Data.conv_stride_size, activation='relu')(embedded_sequences)
        x = MaxPooling1D(Data.conv_pooling_size)(x)
        x = Dropout(Data.conv_dropout_rate)(x)
        x = Conv1D(Data.convolutional_units, Data.conv_stride_size, activation='relu')(x)
        x = MaxPooling1D(Data.conv_pooling_size)(x)
        x = Dropout(Data.conv_dropout_rate)(x)
        x = Conv1D(Data.convolutional_units, Data.conv_stride_size, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(Data.conv_dropout_rate)(x)
        x = Dense(Data.dense_units, activation='relu')(x)

        preds = Dense(len(Data.labels[1]), activation='sigmoid')(x)

        model = Model(sequence_input, preds)

        
        return model