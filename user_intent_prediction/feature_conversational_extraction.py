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

from feature_extraction import check_feature_vector


# -----------------------------------------------------------------#
# -------  conversational features --------------------------------#
# -----------------------------------------------------------------#

# function for: conversation_similarity_feature
def preprocess_utterance_text(utterance):
    processed_utterance = helper.clean_str(utterance)
    processed_utterance = helper.remove_stopwords(processed_utterance) 

    return processed_utterance


# function for: conversation_similarity_feature
def calculate_similarity_score(current_utterance, compared_utterances, vectorizer):

    processed_current_utterance = [preprocess_utterance_text(current_utterance)]
    processed_compared_utterance = [preprocess_utterance_text(compared_utterances)]
    current_utterance_vec = vectorizer.transform(processed_current_utterance)
    compared_utterances_vec = vectorizer.transform(processed_compared_utterance)
     
    similarity_score = cosine_similarity(current_utterance_vec, compared_utterances_vec)[0][0]
    return similarity_score

def conversation_similarity_feature(user_utterance_dataframe, whole_utterance_dataframe):


    whole_utterance_text = whole_utterance_dataframe[['text']].values
    utterances_text = [helper.clean_str(text[0]) for text in whole_utterance_text]
    utterances_text = [helper.remove_stopwords(text) for text in utterances_text]
        
    vectorizer = CountVectorizer()
    vectorizer.fit(utterances_text)

    conversation_similarity_feature_v = []

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


        conversation_similarity_feature_v.append(copy.deepcopy(utterance_feat_vector))


    return conversation_similarity_feature_v # 4 features






# function for: conversation_previous_one_turn_label
def construct_label_vector(u_prev_label_vector, previous_label, label_indext_dict):

    # print(label_indext_dict)
    for label in previous_label:
        label_index = label_indext_dict[label]
        u_prev_label_vector[label_index] = 1

    return u_prev_label_vector
     

# previous_one_turn_features
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


def construct_multi_turn_label_vector(multi_turn_label_vector, previous_label_list, label_indext_dict):
    for labels in previous_label_list:
        for label in labels:
            label_index = label_indext_dict[label]
            multi_turn_label_vector[label_index] += 1

    return multi_turn_label_vector


def conversation_multi_turn_label_features(user_utterance_dataframe, whole_utterance_dataframe, num_previous_turns, intent_label_indext_dict, action_label_indext_dict):

    previous_multi_turn_label_feature_v= []

    for index, user_utterance_data in user_utterance_dataframe.iterrows(): 
    
        u_prev_label_vector = []
        u_prev_intent_label_vector = [0] * len(intent_label_indext_dict)
        u_prev_action_label_vector = [0] * len(action_label_indext_dict)

        # check whether it is the initial sentence
        pos = user_utterance_data ['pos']
        # print('current pos of utterance:', pos)
        if pos == 1:
            pass
        # if not initial sentence
        else:
            # 1. find which conversation it belongs to and find the corresponding conversation
            conversation_id = user_utterance_data ['conversation_id']
            current_conversation_df = whole_utterance_dataframe[whole_utterance_dataframe['conversation_id'] == conversation_id]
            current_conversation_seeker_df = current_conversation_df[current_conversation_df['role'] == 'seeker']
            current_conversation_recommender_df = current_conversation_df[current_conversation_df['role'] == 'recommender']

            # print(current_conversation_df)
            # print(current_conversation_seeker_df)
            # print(current_conversation_recommender_df)


            # 2. find previous utterances/response in the conversation for the utterance
            previous_user_intents_list = []

            # print(current_conversation_seeker_df['pos'].values)
            # print(current_conversation_seeker_df['pos'].values[::-1])
           
            for pos_user_utterance in current_conversation_seeker_df['pos'].values[::-1]:
                if len(previous_user_intents_list) >= num_previous_turns:
                    break
                if pos_user_utterance < pos:
                    # print(pos_user_utterance)
                    previous_user_intents_list.append(current_conversation_df.loc[current_conversation_df.index[pos_user_utterance-1], 'labels'])

            u_prev_intent_label_vector = construct_multi_turn_label_vector(u_prev_intent_label_vector, previous_user_intents_list , intent_label_indext_dict)
            
            # print(u_prev_intent_label_vector)

            previous_recommender_acts_list = []
            for pos_recommender_resp in current_conversation_recommender_df['pos'].values[::-1]:
                if len(previous_recommender_acts_list) >= num_previous_turns:
                    break
                if pos_recommender_resp < pos:
                    # print(pos_recommender_resp)
                    previous_recommender_acts_list.append(current_conversation_df.loc[current_conversation_df.index[pos_recommender_resp-1], 'labels'])
 
            u_prev_action_label_vector = construct_multi_turn_label_vector(u_prev_action_label_vector, previous_recommender_acts_list , action_label_indext_dict)


        # u_prev_label_vector =  copy.deepcopy (u_prev_intent_label_vector + u_prev_action_label_vector)
        u_prev_label_vector =  copy.deepcopy (u_prev_action_label_vector)
        
        previous_multi_turn_label_feature_v.append(copy.deepcopy(u_prev_label_vector))
    
    helper.print_current_time()
    print("The shape of previous multi turn label feature vector : (%d, %d)" % (len(previous_multi_turn_label_feature_v), len(previous_multi_turn_label_feature_v[1])))
    
    return previous_multi_turn_label_feature_v









def extract_conversational_features(user_utterance_dataframe, whole_utterance_dataframe, num_previous_turns, intent_label_indext_dict, action_label_indext_dict):


    conversational_features = [[] for i in range(len(user_utterance_dataframe))]
 
    
    f_conversation_pos = user_utterance_dataframe[['pos']].values.reshape(-1).tolist()
    f_conversation_similarity = conversation_similarity_feature(user_utterance_dataframe, whole_utterance_dataframe)
    if num_previous_turns > 0:
        f_conversation_multi_turn_label = conversation_multi_turn_label_features(user_utterance_dataframe, whole_utterance_dataframe, num_previous_turns, intent_label_indext_dict, action_label_indext_dict)
        conversational_features = np.column_stack((f_conversation_pos,  f_conversation_similarity, f_conversation_multi_turn_label))
        check_feature_vector(f_conversation_multi_turn_label, 'f_conversation_multi_turn_label')
    else:
        conversational_features = np.column_stack((f_conversation_pos,  f_conversation_similarity))
    
    # check_feature_vector
    check_feature_vector(f_conversation_pos, 'f_conversation_pos')
    check_feature_vector(f_conversation_similarity, 'f_conversation_similarity')
    check_feature_vector(conversational_features, 'conversational_features')

    return conversational_features
    












