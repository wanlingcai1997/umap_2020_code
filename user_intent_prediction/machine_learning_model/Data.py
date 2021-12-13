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
    discourse_features = 1
    sentiment_features = 1
    conversational_features = 1
    
    num_previous_turns = 1
    
    # feature normalization
    feature_normalization = 0
        
    # -----------------------------------------------------------------------------------------------
    # ------                         Model Related Configuration                               ------                                 
    # -----------------------------------------------------------------------------------------------


	## model selection 
    neural_model = 1  # 1 denotes it is an neural model, 0 otherwise
    algorithm_adaption = 1 # 1 denotes it is an algorithm_adaption method, 0 otherwise
    problem_transformation = 1 # 1 denotes it is a problem_transformation method, 0 otherwise
    problem_transformation_method = 'CC' #  3 classes of problem_transformation_method : Binary Relevance (BR), Label Powerset (LP), Classifier Chains (CC)
    
    model_name = 'ML_kNN'
    
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
   

