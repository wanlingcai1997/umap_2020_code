import numpy as np
import pandas as pd
from Data import Data
import copy
import sys 
sys.path.append("..") 
import load_data
import feature_extraction
import feature_conversational_extraction
import feature_recommendation_task_extraction
import helper

class ConstructFeature(object):

    @staticmethod
    def constructFeature():
       
        # Prepared Feature Vector From Extracted Conversations
        # (1) Utterance Features (from the perspective of Seeker)
        # (2) Response Features (from the perspective of Recommender)
        # (3) Conversational Features
        # (4) Recommendation Task-Specific Features 

        Data.feature_vector = []
        fitted_vectorizer = None
        entity_label_set = None
        POS_tag_set = None
        first_level_intent_dict = None
        whole_utterance_text = Data.whole_utterance_dataframe[['text']].values.reshape(-1).tolist()
        if Data.content_features == 1:
            fitted_vectorizer = feature_extraction.initialize_fitted_vectorizer (Data.whole_utterance_dataframe)
            entity_label_set = feature_extraction.get_occurred_entity_label_set(whole_utterance_text) 
        if Data.discourse_features == 1:
            POS_tag_set = feature_extraction.get_occurred_POS_tag_set(whole_utterance_text) 
        if Data.task_specific_features == 1:
            first_level_intent_dict = feature_recommendation_task_extraction.get_first_level_intent_dict(Data.higher_level_intent_dict_file)

        for each_conversation_info in Data.conversation_previous_N_turns:
            # Step 1: Extract Seekers' Utterance and Recommender' Responses

            Data.utterance_dataframe = load_data.load_utterance_data_from_extracted_conv(each_conversation_info)
            Data.user_utterance_dataframe  = Data.utterance_dataframe [Data.utterance_dataframe['role'] == 'seeker']
            Data.recommender_response_dataframe  = Data.utterance_dataframe [Data.utterance_dataframe['role'] == 'recommender']

            # print(Data.utterance_dataframe)
            # print(Data.user_utterance_dataframe)
            # print(Data.recommender_response_dataframe)
            
            
            # Step 2: Extract Utterance/Response Features [sentence-level features (1,2)]  
            user_utterance_feature_vector_avg = []
            recommender_response_feature_vector_avg = []

            user_utterance_feature_vector_list = []
            recommender_response_feature_vector_list = []

            if len(Data.user_utterance_dataframe) > 0:
                if Data.content_features == 1 or Data.discourse_features == 1 or Data.sentiment_features == 1:

                    user_utterance_feature_vector_list = feature_extraction.extract_features(Data.user_utterance_dataframe, fitted_vectorizer, entity_label_set, POS_tag_set,  Data.content_features, Data.discourse_features,Data.sentiment_features, Data.pos_file, Data.neg_file)
                    user_utterance_feature_vector_avg = np.mean( np.array(user_utterance_feature_vector_list), axis=0) 
            
                    assert (len(user_utterance_feature_vector_avg) == len(user_utterance_feature_vector_list[0]))
                    
            if len(Data.recommender_response_dataframe) > 0:
                if Data.content_features == 1 or Data.discourse_features == 1 or Data.sentiment_features == 1:
                    recommender_response_feature_vector_list = feature_extraction.extract_features(Data.recommender_response_dataframe, fitted_vectorizer,  entity_label_set, POS_tag_set, Data.content_features, Data.discourse_features,Data.sentiment_features, Data.pos_file, Data.neg_file)
                    recommender_response_feature_vector_avg = np.mean( np.array(recommender_response_feature_vector_list), axis=0) 
                    assert (len(recommender_response_feature_vector_avg) == len(recommender_response_feature_vector_list[0]))
            
           
     

            # Step 3: Extract Conversational label Features [Conversation-level features (1,2)]  
            user_intent_feature_vector = []
            recommender_action_feature_vector = []
 
            if len(Data.user_utterance_dataframe) > 0:
            
                if Data.conversational_features == 1:
                    user_intent_feature_vector = feature_conversational_extraction.extract_multi_turn_label_features(Data.user_utterance_dataframe, Data.intent_label_indext_dict)
            
            if len(Data.recommender_response_dataframe) > 0:
                if Data.conversational_features == 1:    
                    recommender_action_feature_vector = feature_conversational_extraction.extract_multi_turn_label_features(Data.recommender_response_dataframe, Data.action_label_indext_dict)
                    

            # Step 4: Extract Recommendation Interaction Task-Specific Features [Conversation-level features (1,2)]  
            task_specific_features = []
            if len(Data.user_utterance_dataframe) > 0 and len(Data.recommender_response_dataframe) > 0 and Data.task_specific_features == 1:
                
                task_specific_features = feature_recommendation_task_extraction.extract_task_specific_features(Data.user_utterance_dataframe, Data.recommender_response_dataframe, first_level_intent_dict)
            
            integrated_feature_vector = np.concatenate((user_utterance_feature_vector_avg , recommender_response_feature_vector_avg, user_intent_feature_vector,recommender_action_feature_vector, task_specific_features), axis=None)
            # print(len(integrated_feature_vector))
            assert( len(integrated_feature_vector) == len(user_utterance_feature_vector_avg) + len(recommender_response_feature_vector_avg) + len(user_intent_feature_vector) + len(recommender_action_feature_vector) + len(task_specific_features) )
            Data.feature_vector.append(integrated_feature_vector)

            
        Data.feature_vector = np.array(Data.feature_vector)

        # save features into csv file
        feature_csv_file = "feature_file/pre_turns_%d_content_%d_discourse_%d_sentiment_%d_conversation_%d_task_%d_feat.csv" % (Data.num_previous_turns, Data.content_features, Data.discourse_features,Data.sentiment_features, Data.conversational_features, Data.task_specific_features)
        np.savetxt(feature_csv_file, Data.feature_vector , delimiter= ",", fmt='%1.4e')
        # exit()
        #------------------------------------------------            
        helper.print_current_time()
        print("Extract Features For Each extracted Conversations (N utterances) and Construct Feature Vector")

        print("[ %d conversations with the size of features vector (%d) ]" % (len (Data.feature_vector) , len (Data.feature_vector[0]) ))
        #------------------------------------------------            
        
        # save features into csv file
        feature_csv_file = "feature_file/pre_turns_%d_content_%d_discourse_%d_sentiment_%d_conversation_%d_task_%d_feat.csv" % (Data.num_previous_turns, Data.content_features, Data.discourse_features,Data.sentiment_features, Data.conversational_features, Data.task_specific_features)
        np.savetxt(feature_csv_file, Data.feature_vector , delimiter= ",", fmt='%1.4e')
        # exit()
        

         