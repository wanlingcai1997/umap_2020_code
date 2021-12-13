import re
import copy
import numpy as np
import json
import helper
import sys


def get_first_level_intent_dict(first_level_intent_dict_file):
    first_level_intent_dict = {}
    with open(first_level_intent_dict_file, "r") as file:
        first_level_intent_dict = json.load(file)

    print(first_level_intent_dict)

    return first_level_intent_dict


def task_engagement_first_level_intent(user_utterances, first_level_intent_dict):
    engagement_first_level_intent_feat = []

    first_level_intent = ['RF', 'RQ', 'FD']
    previous_turn_label_list = user_utterances[['labels']].values.reshape(-1).tolist()
    
    first_level_intent_freq = {}
    num_label = 0
    for labels in previous_turn_label_list:
        num_label += len(labels)
        for label in labels:
            label_to_higher_level = first_level_intent_dict[label]
            # print(label)
            # print(label_to_higher_level)
            if label_to_higher_level in first_level_intent_freq:
                intent_num = first_level_intent_freq[label_to_higher_level] + 1
                first_level_intent_freq[label_to_higher_level] = intent_num
            else:
                first_level_intent_freq[label_to_higher_level] = 1

    for each_intent in first_level_intent:
        if each_intent in first_level_intent_freq:
            engagement_first_level_intent_feat.append(first_level_intent_freq[each_intent] / num_label)
        else:
            engagement_first_level_intent_feat.append(0)

    return engagement_first_level_intent_feat

# # unfinished
# def task_feedback_frequency(user_utterances):
#     task_feedback_frequency = []
#     return task_feedback_frequency



def task_movie_overlap_absolute_and_normalized(user_utterances, recommender_responses):
    task_movie_overlap_absolute_and_normalized_feat = []
    
    movie_list_by_user = []
    movie_list_by_recommender = []
    user_utterances_text = user_utterances[['text']].values.reshape(-1).tolist()
    recommender_responses_text = recommender_responses[['text']].values.reshape(-1).tolist()
    for utterance in user_utterances_text:
        movie_list_by_user = re.findall(r'@[0-9]+', str(utterance))
        # if (len(movie_id_list)):
        #     for movie_id in movie_id_list:
        #         print(movie_id)
    for utterance in recommender_responses_text:
        movie_list_by_recommender = re.findall(r'@[0-9]+', str(utterance))    

    movie_overlap_absolute = len(set(movie_list_by_user).intersection(set(movie_list_by_recommender)))
    union_movies_list =  len(set(movie_list_by_user).union(set(movie_list_by_recommender)))

    movie_overlap_normalized = 0
    if union_movies_list > 0:
        movie_overlap_normalized = movie_overlap_absolute / union_movies_list

    # print(movie_list_by_user)
    # print(movie_list_by_recommender)

    task_movie_overlap_absolute_and_normalized_feat = [movie_overlap_absolute, movie_overlap_normalized]
    # print(task_movie_overlap_absolute_and_normalized_feat)
    return task_movie_overlap_absolute_and_normalized_feat


def extract_task_specific_features(user_utterances, recommender_responses, first_level_intent_dict):
    task_specific_features = []

    f_task_engagement_first_level_intent = task_engagement_first_level_intent(user_utterances, first_level_intent_dict)
    # f_task_feedback_frequency = task_feedback_frequency(user_utterances)
    f_task_movie_overlap_absolute_and_normalized = task_movie_overlap_absolute_and_normalized(user_utterances, recommender_responses)


    
    task_specific_features = np.concatenate((f_task_engagement_first_level_intent , f_task_movie_overlap_absolute_and_normalized), axis=None)
            
    
    # check feature vector
    # check_feature_vector(f_task_engagement_first_level_intent, 'f_task_engagement_first_level_intent')
    # check_feature_vector(f_task_movie_overlap_absolute_and_normalized, 'f_task_movie_overlap_absolute_and_normalized')
    # # check_feature_vector(f_content_num_movies, 'f_content_num_movies')
    # check_feature_vector(task_specific_features, 'task_specific_features')

    return task_specific_features

    
def check_feature_vector(feature_vector, featur_name):
    helper.print_current_time()
    num_features = len(feature_vector[0]) if isinstance(feature_vector[0],list)  else 1
    print(" %s - Shape: (%d, %d)" % (featur_name, len(feature_vector),num_features))
     