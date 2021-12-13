import numpy as np
import pandas as pd
from Data import Data
import copy
import sys 
sys.path.append("..") 
import feature_extraction
import feature_conversational_extraction
import helper

class ConstructFeature(object):

    @staticmethod
    def constructFeature():


       
        # Prepared Extracted Features 
        # utterance-level features
        if Data.content_features == 1 or Data.structural_features == 1 or Data.sentiment_features == 1 :
            utterance_level_feature_vector = feature_extraction.extract_features(Data.user_utterance_dataframe, Data.content_features, Data.structural_features,Data.sentiment_features, Data.pos_file, Data.neg_file)
        
        # dialogue-level features
        if Data.conversational_features == 1:
            conversational_feature_vector = feature_conversational_extraction.extract_conversational_features(Data.user_utterance_dataframe, Data.whole_utterance_dataframe)
            if Data.content_features == 1 or Data.structural_features == 1 or Data.sentiment_features == 1 :
                Data.feature_vector = np.array(np.column_stack((utterance_level_feature_vector, conversational_feature_vector))).astype('float64')
            else: 
                Data.feature_vector = np.array(conversational_feature_vector)

        else:
            Data.feature_vector = utterance_level_feature_vector
        #------------------------------------------------            
        helper.print_current_time()
        print("Extract Features For Each Utternaces and Construct Feature Vector")

        print("[ %d utterances with the size of features vector (%d) " % (Data.feature_vector.shape[0],Data.feature_vector.shape[1]))
        #------------------------------------------------            
        
       
        

         