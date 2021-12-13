import numpy as np
import pandas as pd
from Data import Data
import copy
import sys 
from sklearn.model_selection import train_test_split

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
sys.path.append("..") 
import generate_word_embedding
import helper

class Initialization(object):

    @staticmethod
    def initialize():

        # preprocessing the utterances
        utterances = Data.user_utterance_dataframe[['text']].values
        utterances_text = [helper.clean_str(text[0]) for text in utterances]
        utterances_text = [helper.remove_stopwords(text) for text in utterances_text]
        
        # text vectorization -> into a 2D integer tensor
        tokenizer = Tokenizer(num_words=Data.max_num_vocabulary)
        tokenizer.fit_on_texts(utterances_text)  
        
        sequences = tokenizer.texts_to_sequences(utterances_text) # convert each utterance into a sequence of word vectors.

        # 
        actual_max_seq_len = max(len(seq) for seq in sequences)
        Data.final_max_seq_len = min(actual_max_seq_len, Data.max_sequence_length)  
        
        #------------------------------------------------
        helper.print_current_time() 
        print("actual_max_seq_len:" , actual_max_seq_len)
        helper.print_current_time()  
        print("final_max_seq_len:" , Data.final_max_seq_len)
        #------------------------------------------------

        # pads sequences to the same length (final_max_seq_len).
        Data.u_sequence_data = pad_sequences(sequences, maxlen=Data.final_max_seq_len, padding='post')
        #------------------------------------------------
        helper.print_current_time() 
        print("after padding sequences, we obtained the text sequence. Shape:" , Data.u_sequence_data.shape)
        #------------------------------------------------


        #------------------------------------------------
        # incorporate other features
        #------------------------------------------------



        #------------------------------------------------
        # prepare embedding matrix
        word_index = tokenizer.word_index
        Data.actual_num_of_vocabulary = min(Data.max_num_vocabulary, len(word_index) + 1)
        #------------------------------------------------
        helper.print_current_time()  
        print('Found %s unique tokens (actual_num_of_vocabulary).' % len(word_index))
        #------------------------------------------------
        #------------------------------------------------
        helper.print_current_time()  
        print('Constructing embedding matrix ...')
        helper.print_current_time()  
        print('The dimension of embeeding:', Data.embedding_dimension)
        #------------------------------------------------

        if Data.embedding_selection == 'glove':
            Data.embedding_matrix = generate_word_embedding.create_word_embedding_matrix(Data.embedding_dimension, Data.actual_num_of_vocabulary, word_index)
            
        if Data.embedding_selection == 'word2vec':
            pass

        #------------------------------------------------
        helper.print_current_time()  
        print('The length of embedding matrix:',len(Data.embedding_matrix))
        #------------------------------------------------


        #------------------------------------------------
        #------------------------------------------------
        #------------------------------------------------
        #------------  Dataset Split --------------------
        #------------------------------------------------
        #------------------------------------------------
        #------------------------------------------------

        # Split dataset into Train, Test and Validation set
        Data.x_train, x_test_val, Data.y_train, y_test_val = train_test_split(Data.u_sequence_data , Data.labels, test_size=0.2, random_state=42)
        Data.x_val, Data.x_test, Data.y_val, Data.y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42)
    
        print(Data.x_train)
     
        assert len(Data.x_train) + len(Data.x_val) + len(Data.x_test) == len(utterances)
        assert len(Data.y_train) + len(Data.y_val) + len(Data.y_test) == len(utterances)
  
        

        return 