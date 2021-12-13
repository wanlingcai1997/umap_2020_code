import numpy as np
import pandas as pd
import json
import helper
import copy

def load_whole_annotation_data_dict(file):  

    whole_annotation_data_dict = {}    # whole data dict
    # read whole annotation json file
    with open(file, "r", encoding ='utf-8') as file:
        whole_annotation_data_dict = json.load(file)

    helper.print_current_time()
    print("-- Load Whole Annotation Data -- Number of Conversation: " +  str(len(whole_annotation_data_dict)))
    return whole_annotation_data_dict


def load_utterance_data(whole_annotation_data_dict):
    whole_utterance_conversation_id = []
    whole_utterance_text = []
    whole_utterance_role = []
    whole_utterance_pos = []
    whole_utterance_pos_normalized = []
    whole_utterance_labels = []

    intent_labels = []
    action_labels = []

    for conversation_id in whole_annotation_data_dict:
        conversation_info = whole_annotation_data_dict[conversation_id]['conversation_info']
        for utterance_key in conversation_info:
            utterance_info = conversation_info[utterance_key]
            whole_utterance_conversation_id.append(copy.deepcopy(conversation_id))
            whole_utterance_text.append(utterance_info['utterance_text'])
            whole_utterance_role.append(utterance_info['role'])
            whole_utterance_pos.append(utterance_info['utterance_pos'])
            whole_utterance_pos_normalized.append(utterance_info['utterance_pos']/len(conversation_info))
            whole_utterance_labels.append(utterance_info['intent/action'])
        
            for label in utterance_info['intent/action']:
        
                if utterance_info['role'] == 'seeker' and label not in intent_labels:
                    if label == None:
                        print (utterance_info['utterance_text'])
                    intent_labels.append(label)
                if utterance_info['role'] == 'recommender' and label not in action_labels:
                    action_labels.append(label)

    whole_utterance_df = pd.DataFrame({'conversation_id': whole_utterance_conversation_id,'text':whole_utterance_text, 'role':whole_utterance_role, 'pos':whole_utterance_pos, 'normalized_pos': whole_utterance_pos_normalized, 'labels':whole_utterance_labels})
    helper.print_current_time()
    print("-- Load Whole Utterance Data  -- Number of Utterances: " +  str(whole_utterance_df.shape[0]))
    return whole_utterance_df, intent_labels, action_labels

def construct_label_index_dict (labels):
    label_indext_dict = {}
    indext_label_dict = {}
    index = 0
    for label in labels:
        label_indext_dict[label] = index
        indext_label_dict[index] = label
        index += 1
        
    return label_indext_dict, indext_label_dict