# Data.py

import numpy as np


class Data(object):
    # ===============================================================================================
    # ====================          Configuration Data      =========================================                          
    # ===============================================================================================                          

    
    # -----------------------------------------------------------------------------------------------
    # ------                                   Data                                            ------
    # -----------------------------------------------------------------------------------------------

    file_input_data = None

    # -----------------------------------------------------------------------------------------------
    # ------                       Feature Related Configuration                               ------                                 
    # -----------------------------------------------------------------------------------------------
    
    # feature construction
    content_features = 1
    structural_features = 1
    sentiment_features = 1
    conversational_features = 1
    
    # feature normalization
    feature_normalization = 0
        
    # -----------------------------------------------------------------------------------------------
    # ------                         Model Related Configuration                               ------                                 
    # -----------------------------------------------------------------------------------------------


	## model selection 
    model_name = 'BiLSTM'

    max_sequence_length = 50
    max_num_vocabulary = 20000
    embedding_selection = 'glove' # word2vec or glove'
    embedding_dimension = 100


    lstm_units = 512
    dropout_rate = 0.6
    dense_units = 256
    

    #  --------------------------------- Model Parameters -------------------------------------------


    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # ------                                    Evaluation                                     ------                                 
    # -----------------------------------------------------------------------------------------------

    ## cross-validation
    cross_validation = 3

    
    
    # ===============================================================================================
    # ====================         Additional Data Resource    ======================================                          
    # ===============================================================================================                          

    # feature data
    pos_file = '../data/positive-words.txt'
    neg_file = '../data/negative-words.txt'

    # glove_embedding_file = '' (specify in the corresponding function)

    # ===============================================================================================
    # ====================      Constructed Data For Model       ====================================                          
    # ===============================================================================================                          


    # -----------------------------------
    # ------    Utterance Data     ------                                 
    # -----------------------------------
    whole_annotation_data_dict = None
    whole_utterance_dataframe = None
    user_utterance_dataframe = None

    # -----------------------------------
    # ------      Label Data       ------                                 
    # -----------------------------------
    intent_labels = None
    action_labels = None

    intent_label_indext_dict = None
    intent_indext_label_dict = None
    action_label_indext_dict = None
    action_indext_label_dict = None

    labels = None # one-hot
    


    # -----------------------------------
    # ------ Vector Representation ------                                 
    # -----------------------------------
    feature_vector = None
    u_sequence_data = None

    embedding_matrix = None

    # -----------------------------------
    # --- Model Parameter (Actual) ------                                 
    # -----------------------------------
    final_max_seq_len = 50
    actual_num_of_vocabulary = 20000

    # -----------------------------------
    # ------- Dataset Preparation -------                                 
    # -----------------------------------
    x_train = None
    x_test = None
    x_val = None
    y_train = None
    y_test = None
    y_val = None
    
    # -----------------------------------
    # ------- After Training ------------                                 
    # -----------------------------------
    trained_model = None


