import re
from nltk import word_tokenize 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import helper
import copy
import sys
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------#
# -------  conversational features --------------------------------#
# -----------------------------------------------------------------#
def preprocess_utterance_text(utterance):
    processed_utterance = helper.clean_str(utterance)
    processed_utterance = helper.remove_stopwords(processed_utterance) 

    return processed_utterance

def calculate_similarity_score(current_utterance, compared_utterances, vectorizer):

    processed_current_utterance = [preprocess_utterance_text(current_utterance)]
    processed_compared_utterance = [preprocess_utterance_text(compared_utterances)]
    current_utterance_vec = vectorizer.transform(processed_current_utterance)
    compared_utterances_vec = vectorizer.transform(processed_compared_utterance)
     
    similarity_score = cosine_similarity(current_utterance_vec, compared_utterances_vec)[0][0]
    return similarity_score


def extract_conversational_features(user_utterance_dataframe, whole_utterance_dataframe):

    whole_utterance_text = whole_utterance_dataframe[['text']].values
    utterances_text = [helper.clean_str(text[0]) for text in whole_utterance_text]
    utterances_text = [helper.remove_stopwords(text) for text in utterances_text]
        
    vectorizer = CountVectorizer()
    vectorizer.fit(utterances_text)


    utterances_conversational_feature_vector = []

    for index, user_utterance_data in user_utterance_dataframe.iterrows():
        # conversational feature
        utterance_feat_vector = []
        similarity_with_initial_user_utterance = 1
        similarity_with_previous_user_utterance = 1
        similarity_with_previous_sys_responses = 0
        
        similarity_with_previous_interactions = 1

        # obtain utterance text
        current_utterance_text = user_utterance_data['text']

        # check whether it is the initial sentence
        pos = user_utterance_data ['pos']
        if pos == 1:
            utterance_feat_vector.append(similarity_with_initial_user_utterance)
            utterance_feat_vector.append(similarity_with_previous_user_utterance)
            utterance_feat_vector.append(similarity_with_previous_sys_responses)
            utterance_feat_vector.append(similarity_with_previous_interactions)

        # if not initial sentence
        else:
            # 1. find which conversation it belongs to and find the corresponding conversation
            conversation_id = user_utterance_data ['conversation_id']
            current_conversation_df = whole_utterance_dataframe[whole_utterance_dataframe['conversation_id'] == conversation_id]
            # 2. find initial utterance for the conversation (calculate the similarity)
           
            if current_conversation_df.loc[current_conversation_df.index[0],'role'] == 'seeker':
                initial_utterance = current_conversation_df.loc[current_conversation_df.index[0], 'text']
            else:
                initial_utterance = current_conversation_df.loc[current_conversation_df.index[1],'role']

            similarity_with_initial_user_utterance = calculate_similarity_score(current_utterance_text, initial_utterance, vectorizer)
            # print(similarity_with_initial_user_utterance)
           
            # 3. find previous utterances/response in the conversation for the utterance (calculate the similarity)
            if pos == 2:
                previous_user_utterance = current_conversation_df.loc[current_conversation_df.index[1], 'text']
            else:
                previous_user_utterance = current_conversation_df.loc[current_conversation_df.index[pos-2-1], 'text']
            similarity_with_previous_user_utterance = calculate_similarity_score(current_utterance_text, previous_user_utterance, vectorizer)
            # print(similarity_with_previous_user_utterance)
        
            previous_system_response = current_conversation_df.loc[current_conversation_df.index[pos-1-1], 'text']
            similarity_with_previous_sys_responses = calculate_similarity_score(current_utterance_text, previous_system_response, vectorizer)
            # print(similarity_with_previous_sys_responses)

            # 4. aggregate all the previous interaction text (calculate the similarity)
            previous_interactions = ''
            for i in range(0, pos-1):
                previous_interactions += current_conversation_df.loc[current_conversation_df.index[i], 'text']
                
            similarity_with_previous_interactions = calculate_similarity_score(current_utterance_text, previous_interactions, vectorizer)
            # print(similarity_with_previous_interactions)

            utterance_feat_vector.append(similarity_with_initial_user_utterance)
            utterance_feat_vector.append(similarity_with_previous_user_utterance)
            utterance_feat_vector.append(similarity_with_previous_sys_responses)
            utterance_feat_vector.append(similarity_with_previous_interactions)


        utterances_conversational_feature_vector.append(copy.deepcopy(utterance_feat_vector))


    return utterances_conversational_feature_vector # 4 features





def construct_label_vector(u_prev_label_vector, previous_label, label_indext_dict):

    # print(label_indext_dict)
    for label in previous_label:
        label_index = label_indext_dict[label]
        u_prev_label_vector[label_index] = 1

    return u_prev_label_vector
     

def extract_previous_round_features(user_utterance_dataframe, whole_utterance_dataframe, intent_label_indext_dict, action_label_indext_dict):
    

    previous_label_feature_vector = []
    
    
    for index, user_utterance_data in user_utterance_dataframe.iterrows(): 
        u_prev_label_vector = []
        u_prev_intent_label_vector = [0] * len(intent_label_indext_dict)
        u_prev_action_label_vector = [0] * len(action_label_indext_dict)

        # check whether it is the initial sentence
        pos = user_utterance_data ['pos']
        if pos == 1:
            pass
            # u_prev_label_vector = copy.deepcopy (u_prev_intent_label_vector + u_prev_action_label_vector)
        # if not initial sentence
        else:
            # 1. find which conversation it belongs to and find the corresponding conversation
            conversation_id = user_utterance_data ['conversation_id']
            current_conversation_df = whole_utterance_dataframe[whole_utterance_dataframe['conversation_id'] == conversation_id]
            # 2. find previous utterances/response in the conversation for the utterance
            if pos == 2:
                previous_user_intents = []
            else:
                previous_user_intents = current_conversation_df.loc[current_conversation_df.index[pos-2-1], 'labels']
                # print(previous_user_intents)

            u_prev_intent_label_vector = construct_label_vector(u_prev_intent_label_vector, previous_user_intents , intent_label_indext_dict)
            # print(u_prev_intent_label_vector)
        
            previous_system_act = current_conversation_df.loc[current_conversation_df.index[pos-1-1], 'labels']
            # print(previous_system_act)
            u_prev_action_label_vector = construct_label_vector(u_prev_action_label_vector, previous_system_act , action_label_indext_dict)
            
            # print(u_prev_action_label_vector)
        u_prev_label_vector = copy.deepcopy (u_prev_action_label_vector)
            # u_prev_label_vector = copy.deepcopy (u_prev_intent_label_vector + u_prev_action_label_vector)
            # print(u_prev_label_vector)

        previous_label_feature_vector.append(copy.deepcopy(u_prev_label_vector))
        # print(u_prev_label_vector)
    return previous_label_feature_vector


def construct_multi_turn_system_act_features_vector(multi_turn_system_act_features_vector, previous_label_list, action_label_indext_dict):

    for labels in previous_label_list:
        for label in labels:
            label_index = action_label_indext_dict[label]
            multi_turn_system_act_features_vector[label_index] += 1

    return multi_turn_system_act_features_vector



# for user intent prediction
def extract_multi_turn_system_act_features(user_utterance_dataframe, whole_utterance_dataframe, action_label_indext_dict):

    previous_multi_turn_system_act_features = []

    for index, user_utterance_data in user_utterance_dataframe.iterrows(): 
    
        u_prev_multi_turn_system_act_vector = [0] * len(action_label_indext_dict)

        # check whether it is the initial sentence
        pos = user_utterance_data ['pos']
        if pos == 1:
            pass
        # if not initial sentence
        else:
            # 1. find which conversation it belongs to and find the corresponding conversation
            conversation_id = user_utterance_data ['conversation_id']
            current_conversation_df = whole_utterance_dataframe[whole_utterance_dataframe['conversation_id'] == conversation_id]
            current_conversation_system_act_df = current_conversation_df[current_conversation_df['role'] == 'recommender']
            # print(current_conversation_system_act_df)
            # 2. find previous system response in the conversation 
            previous_system_act = []
            for pos_system_resp in current_conversation_system_act_df['pos'].values:
                if pos_system_resp < pos:
                    previous_system_act.append(current_conversation_system_act_df.loc[current_conversation_df.index[pos_system_resp-1], 'labels'])
            # print(previous_system_act)
            
            u_prev_multi_turn_system_act_vector = construct_multi_turn_system_act_features_vector(u_prev_multi_turn_system_act_vector, previous_system_act , action_label_indext_dict)
            
            # print(u_prev_multi_turn_system_act_vector)

        previous_multi_turn_system_act_features.append(copy.deepcopy(u_prev_multi_turn_system_act_vector))
    
    return previous_multi_turn_system_act_features



# for user satisfaction prediction
def extract_multi_turn_label_features(utterance_dataframe, label_indext_dict):
    previous_turn_label_list = utterance_dataframe[['labels']].values.reshape(-1).tolist()
    prev_multi_turn_label_vector = [0] * len(label_indext_dict)
    # print(previous_turn_label_list)
    for labels in previous_turn_label_list:
        for label in labels:
            label_index = label_indext_dict[label]
            prev_multi_turn_label_vector[label_index] += 1

    # print(previous_turn_label_list)
    # print(prev_multi_turn_label_vector)
    # check_feature_vector (prev_multi_turn_label_vector, 'prev_multi_turn_label_vector')
    return prev_multi_turn_label_vector


def check_feature_vector(feature_vector, featur_name):
    helper.print_current_time()
    num_features = 1 if type(feature_vector[0]) == int else len(feature_vector[0])
    print(" %s - Shape: (%d, %d)" % (featur_name, len(feature_vector),num_features))
    













