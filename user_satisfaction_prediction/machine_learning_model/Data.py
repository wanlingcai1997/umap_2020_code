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
    num_previous_turns = None
    # -----------------------------------------------------------------------------------------------
    # ------                       Feature Related Configuration                               ------                                 
    # -----------------------------------------------------------------------------------------------
    
    # feature construction
    content_features = 1
    discourse_features = 1
    sentiment_features = 1
    conversational_features = 1
    task_specific_features = 1
    # feature normalization
    feature_normalization = 0
        
    # -----------------------------------------------------------------------------------------------
    # ------                         Model Related Configuration                               ------                                 
    # -----------------------------------------------------------------------------------------------


	## model selection 
   
    model_name = 'LR'
    
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
    pos_file = '../../additional_resource/opinion_lexicon/positive-words.txt'
    neg_file = '../../additional_resource/opinion_lexicon/negative-words.txt'
    higher_level_intent_dict_file = '../../additional_resource/intent_dict/higher_level_intent_dict.json'


    # ===============================================================================================
    # ====================      Constructed Data For Model       ====================================                          
    # ===============================================================================================                          


    # -----------------------------------
    # ------    Conversational Data     ------                                 
    # -----------------------------------
    whole_annotation_data_dict = None
    whole_utterance_dataframe = None
    conversation_previous_N_turns = None
    conversation_labels = None

    utterance_dataframe = None
    user_utterance_dataframe = None
    recommender_response_dataframe = None

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
   

