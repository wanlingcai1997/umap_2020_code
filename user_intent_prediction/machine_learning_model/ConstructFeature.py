from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from Data import Data
import copy
import time
import sys 
sys.path.append("..") 
import feature_extraction
import feature_conversational_extraction
import helper


class ConstructFeature(object):

    @staticmethod
    def constructFeature():


        # previous_round_features = feature_conversational_extraction.extract_previous_round_features(Data.user_utterance_dataframe, Data.whole_utterance_dataframe,  Data.intent_label_indext_dict, Data.action_label_indext_dict)
        # Prepared Extracted Features 
        
        # utterance-level features
        content_feature_v = [[] for i in range(len(Data.user_utterance_dataframe))]
        discourse_feature_v = [[] for i in range(len(Data.user_utterance_dataframe))]
        sentiment_feature_v = [[] for i in range(len(Data.user_utterance_dataframe))]
        conversational_feature_v = [[] for i in range(len(Data.user_utterance_dataframe))]

        if Data.content_features == 1:
            start = time.process_time()
            content_feature_v = feature_extraction.extract_content_features(Data.user_utterance_dataframe)
            end = time.process_time()
            helper.print_current_time()
            print ('Feature Construction: content_features ---- run time : %ss ' % str(end-start))

        if Data.discourse_features == 1:
            start = time.process_time()
            discourse_feature_v = feature_extraction.extract_discourse_features(Data.user_utterance_dataframe)
            end = time.process_time()
            helper.print_current_time()
            print ('Feature Construction: discourse_features ---- run time : %ss ' % str(end-start))

        if Data.sentiment_features == 1:
            start = time.process_time()
            sentiment_feature_v = feature_extraction.extract_sentiment_features(Data.user_utterance_dataframe, Data.pos_file, Data.neg_file)
            end = time.process_time()
            helper.print_current_time()
            print ('Feature Construction: sentiment_features ---- run time : %ss ' % str(end-start))

        # dialogue-level features
        if Data.conversational_features == 1:
            start = time.process_time()
            conversational_feature_v = feature_conversational_extraction.extract_conversational_features(Data.user_utterance_dataframe, Data.whole_utterance_dataframe, \
                Data.num_previous_turns, Data.intent_label_indext_dict, Data.action_label_indext_dict)
            end = time.process_time()
            helper.print_current_time()
            print ('Feature Construction: conversational_features ---- run time : %ss ' % str(end-start))

        Data.feature_vector = np.array(np.column_stack((content_feature_v, discourse_feature_v, sentiment_feature_v, conversational_feature_v))).astype('float64')
        if Data.feature_normalization == 1:
            feature_scaler = MinMaxScaler()
            Data.feature_vector = feature_scaler.fit_transform(Data.feature_vector)
            helper.print_current_time()
            print ("Feature Normalization with MinMaxScaler.")

        #------------------------------------------------            
        helper.print_current_time()
        print("Extract Features For Each Utternaces and Construct Feature Vector [ %d utterances with the size of features vector (%d) !" % (Data.feature_vector.shape[0],Data.feature_vector.shape[1]) )

        #------------------------------------------------            
        
       
        

         