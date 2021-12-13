from sklearn.preprocessing import MinMaxScaler
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

class LoadFeature(object):

    @staticmethod
    def loadFeature():
       
        # Load Features from Stored Feature Files

        feature_csv_file = "feature_file/pre_turns_%d_content_1_discourse_1_sentiment_1_conversation_1_task_1_feat.csv" % (Data.num_previous_turns)
        whole_feature_vector  = np.loadtxt(feature_csv_file, delimiter= ",")
        
        initial_index = 0
        num_content_feature = 5495
        num_discourse_feature = 29
        num_sentiment_feature = 6
        num_all_sentence_feature = num_content_feature + num_discourse_feature + num_sentiment_feature

        num_conv_intent_feature = 16
        num_conv_action_feature = 10
        num_task_specific_feature = 5

        user_content_start = initial_index
        user_content_end = user_content_start + num_content_feature
        user_discourse_start = user_content_end
        user_discourse_end = user_discourse_start + num_discourse_feature
        user_sentiment_start = user_discourse_end
        user_sentiment_end = user_sentiment_start + num_sentiment_feature
        assert ( num_all_sentence_feature == user_sentiment_end)
        rec_content_start = num_all_sentence_feature
        rec_content_end = rec_content_start + num_content_feature
        rec_discourse_start = rec_content_end
        rec_discourse_end = rec_discourse_start + num_discourse_feature
        rec_sentiment_start = rec_discourse_end
        rec_sentiment_end = rec_sentiment_start + num_sentiment_feature
        assert ( num_all_sentence_feature * 2 == rec_sentiment_end)
        conv_intent_start = rec_sentiment_end
        conv_intent_end = conv_intent_start + num_conv_intent_feature
        conv_action_start = conv_intent_end
        conv_action_end = conv_action_start + num_conv_action_feature
        task_specific_start = conv_action_end
        task_specific_end = task_specific_start + num_task_specific_feature
        assert ( num_all_sentence_feature * 2 + num_conv_intent_feature + num_conv_action_feature + num_task_specific_feature == task_specific_end)
        
        content_feature_vector = [[] for i in range(len(Data.conversation_previous_N_turns))]
        discourse_feature_vector = [[] for i in range(len(Data.conversation_previous_N_turns))]
        sentiment_feature_vector = [[] for i in range(len(Data.conversation_previous_N_turns))]
        conversation_feature_vector = [[] for i in range(len(Data.conversation_previous_N_turns))]
        task_specific_feature_vector = [[] for i in range(len(Data.conversation_previous_N_turns))]
        
        if Data.num_previous_turns == 1:
            rec_content_start = initial_index
            rec_content_end = rec_content_start + num_content_feature
            rec_discourse_start = rec_content_end
            rec_discourse_end = rec_discourse_start + num_discourse_feature
            rec_sentiment_start = rec_discourse_end
            rec_sentiment_end = rec_sentiment_start + num_sentiment_feature
            assert ( num_all_sentence_feature == rec_sentiment_end)
            conv_action_start = rec_sentiment_end
            conv_action_end = conv_action_start + num_conv_action_feature
            assert ( num_all_sentence_feature + num_conv_action_feature == conv_action_end)
            if Data.content_features == 1:
                content_feature_vector = whole_feature_vector[:,rec_content_start:rec_content_end]
                print("The shape of content_feature_vector : " , content_feature_vector.shape )
            if Data.discourse_features == 1:
                discourse_feature_vector = whole_feature_vector[:,rec_discourse_start:rec_discourse_end]
                print("The shape of discourse_feature_vector : " , discourse_feature_vector.shape )
            if Data.sentiment_features == 1:
                sentiment_feature_vector = whole_feature_vector[:,rec_sentiment_start:rec_sentiment_end]
                print("The shape of sentiment_feature_vector : " , sentiment_feature_vector.shape )
            if Data.conversational_features == 1:
                conversation_feature_vector = whole_feature_vector[:,conv_action_start:conv_action_end]
                print("The shape of conversation_feature_vector : " , conversation_feature_vector.shape )

        else:
            if Data.content_features == 1:
                user_content_feature_vector = whole_feature_vector[:,rec_content_start:rec_content_end]
                rec_content_feature_vector = whole_feature_vector[:,rec_content_start:rec_content_end]
                content_feature_vector = np.column_stack((user_content_feature_vector, rec_content_feature_vector))
                print("The shape of content_feature_vector : " , content_feature_vector.shape )
            if Data.discourse_features == 1:
                user_discourse_feature_vector = whole_feature_vector[:,user_discourse_start:user_discourse_end]
                rec_discourse_feature_vector = whole_feature_vector[:,rec_discourse_start:rec_discourse_end]
                discourse_feature_vector = np.column_stack((user_discourse_feature_vector, rec_discourse_feature_vector))
                
                print("The shape of discourse_feature_vector : " , discourse_feature_vector.shape )
            if Data.sentiment_features == 1:
                user_sentiment_feature_vector = whole_feature_vector[:,user_sentiment_start:user_sentiment_end]
                rec_sentiment_feature_vector = whole_feature_vector[:,rec_sentiment_start:rec_sentiment_end]
                sentiment_feature_vector = np.column_stack((user_sentiment_feature_vector, rec_sentiment_feature_vector))
                
                print("The shape of sentiment_feature_vector : " , sentiment_feature_vector.shape )
            if Data.conversational_features == 1:
                intent_feature_vector = whole_feature_vector[:,conv_intent_start:conv_intent_end]
                action_feature_vector = whole_feature_vector[:,conv_action_start:conv_action_end]
                conversation_feature_vector = np.column_stack((intent_feature_vector, action_feature_vector))
                print("The shape of conversation_feature_vector : " , conversation_feature_vector.shape )
            if Data.task_specific_features == 1:
                task_specific_feature_vector = whole_feature_vector[:,task_specific_start:task_specific_end]
                
                print("The shape of task_specific_feature_vector : " , task_specific_feature_vector.shape )



       
        Data.feature_vector = np.column_stack((content_feature_vector, discourse_feature_vector, sentiment_feature_vector, conversation_feature_vector, task_specific_feature_vector))
        if Data.feature_normalization == 1:
            feature_scaler = MinMaxScaler()
            Data.feature_vector = feature_scaler.fit_transform(Data.feature_vector)
            helper.print_current_time()
            print ("Feature Normalization with MinMaxScaler.")

        #------------------------------------------------            
        helper.print_current_time()
        print("Extract Features For Each extracted Conversations (N utterances) and Construct Feature Vector")

        print("[ %d conversations with the size of features vector (%d) ]" % (len (Data.feature_vector) , len (Data.feature_vector[0]) ))
        #------------------------------------------------            
        
        