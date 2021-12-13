import numpy as np
import pandas as pd
from Data import Data
import json
import copy
from sklearn.preprocessing import MultiLabelBinarizer
import sys 
sys.path.append("..") 
import load_data
import helper
class ReadData(object):

    @staticmethod
    def readData():
        if Data.file_input_data != None :
            #------------------------------------------------
            helper.print_current_time()
            print("Read Data")
            #------------------------------------------------
    
            # ------------------------------------------------------------------------------------------------------
            # Step 1: Load Data
            # ------------------------------------------------------------------------------------------------------
            Data.whole_annotation_data_dict = load_data.load_whole_annotation_data_dict(Data.file_input_data)            
            Data.whole_utterance_dataframe, Data.intent_labels, Data.action_labels= load_data.load_utterance_data(Data.whole_annotation_data_dict)
            # Store intent/action labels into two dict with index for convenient use 
            Data.intent_label_indext_dict, Data.intent_indext_label_dict = load_data.construct_label_index_dict(Data.intent_labels)
            Data.action_label_indext_dict, Data.action_indext_label_dict = load_data.construct_label_index_dict(Data.action_labels)
            #------------------------------------------------            
            helper.print_current_time()
            print("Intent Labels (%d):" % len(Data.intent_labels), end=' ')
            print(Data.intent_labels)
            helper.print_current_time()
            print("Action Labels (%d):" % len(Data.action_labels), end=' ')
            print(Data.action_labels)
            #------------------------------------------------            

            # ------------------------------------------------------------------------------------------------------
            # Step 2: Split Conversational Data Into 2 Class -> [Satisfied Conversation (with Label "1"); UnSatisfied Conversaiton (with Label "0")]
            # ------------------------------------------------------------------------------------------------------

            NUM_PREVIOUS_TURNS = Data.num_previous_turns

            conversation_num = 0

            Data.conversation_previous_N_turns = []
            Data.conversation_labels = []

            for key, conversation in Data.whole_annotation_data_dict.items():
                current_conversation_info = conversation['conversation_info']
                # For Unsatisfied Conversation
                if len(conversation['Is_recommendation_accepted']) == 0:
                    Data.conversation_labels.append(1)
                    # Extract N turns before "Thanks"
                    Data.conversation_previous_N_turns.append(copy.deepcopy(ReadData.extract_previous_N_turns_for_unsatCon(current_conversation_info, NUM_PREVIOUS_TURNS)))
                # For Satisfied Conversation
                else:
                    Data.conversation_labels.append(0)
                    # Extract N turns before "Accept"
                    first_accept_pos = min(conversation['Is_recommendation_accepted'])
                    # print(conversation['Is_recommendation_accepted'])
                    # print(first_accept_pos)
                    Data.conversation_previous_N_turns.append(copy.deepcopy(ReadData.extract_previous_N_turns_from_pos(current_conversation_info, NUM_PREVIOUS_TURNS, first_accept_pos)))
                    
                conversation_num += 1 
            assert(conversation_num == len(Data.conversation_labels) == len(Data.conversation_previous_N_turns))
            #------------------------------------------------            
            helper.print_current_time()
            print("Number of Satisfactory Conversation: %d." % Data.conversation_labels.count(0))

            helper.print_current_time()
            print("Number of Unsatisfactory Conversation: %d." % Data.conversation_labels.count(1))

            #------------------------------------------------            

          
    @staticmethod
    def extract_previous_N_turns_for_unsatCon(conversation_info, NUM_PREVIOUS_TURNS):
        extracted_conversation = {}
        length_conversation = len(conversation_info)
        # First Task : Locate the "Thanks" (i.e., OTH) Label as unsatisfied sign / 
        # Step 1: Find the utterance with "OTH" for seeker and recommender, respectively
        last_seeker_utterance_pos = -1
        ukeys_with_OTH_Seeker = []
        ukeys_with_OTH_Recommender = []
        for ukeys, utterance_info in conversation_info.items():
            if 'S' in ukeys:
                last_seeker_utterance_pos = int(ukeys[1:])
            if 'OTH' in utterance_info['intent/action']:
                if 'S' in ukeys:
                    ukeys_with_OTH_Seeker.append(ukeys)
                if 'R' in ukeys:
                    ukeys_with_OTH_Recommender.append(ukeys)

        # Step 2: Identify the "Real" unsatisfied sign
        final_unsatisfied_pos = -1 # utterance position from ** Sender **
        seeker_unsatisfied_pos = -1
        recommender_unsatisfied_pos = -1
        if len(ukeys_with_OTH_Seeker) > 0:
            seeker_unsatisfied_pos = int(ukeys_with_OTH_Seeker[-1][1:])
            while int(seeker_unsatisfied_pos-2) in ukeys_with_OTH_Seeker and seeker_unsatisfied_pos > length_conversation/2:
                seeker_unsatisfied_pos = seeker_unsatisfied_pos - 2
            # print(seeker_unsatisfied_pos)
            if seeker_unsatisfied_pos < length_conversation/2:
                seeker_unsatisfied_pos = -1
        
        if len(ukeys_with_OTH_Recommender) > 0:
            recommender_unsatisfied_pos = int(ukeys_with_OTH_Recommender[-1][1:])
            while int(recommender_unsatisfied_pos-2) in ukeys_with_OTH_Recommender and recommender_unsatisfied_pos > length_conversation/2:
                recommender_unsatisfied_pos = recommender_unsatisfied_pos - 2
            if recommender_unsatisfied_pos < length_conversation/2:
                recommender_unsatisfied_pos = -1
        
        if seeker_unsatisfied_pos > 1 :
            final_unsatisfied_pos = seeker_unsatisfied_pos
        elif recommender_unsatisfied_pos > 1:
            final_unsatisfied_pos = recommender_unsatisfied_pos - 1
        else:
            final_unsatisfied_pos = last_seeker_utterance_pos 

        # print(length_conversation)
        # print(final_unsatisfied_pos)
        # print('----')
            
        extracted_conversation = ReadData.extract_previous_N_turns_from_pos(conversation_info, NUM_PREVIOUS_TURNS, final_unsatisfied_pos)
        
        return extracted_conversation

    @staticmethod
    def extract_previous_N_turns_from_pos(conversation_info, NUM_PREVIOUS_TURNS, utterance_pos):
        extracted_conversation = []
        extracted_utterance_pos_list = []
        initial_utterance_pos = 1
        if utterance_pos-NUM_PREVIOUS_TURNS >= 1:
            initial_utterance_pos = utterance_pos-NUM_PREVIOUS_TURNS 
        
        # print(initial_utterance_pos)
        
    
        for i in range(initial_utterance_pos, utterance_pos, 1):
            extracted_utterance_pos_list.append(i)

        for ukeys, utterance_info in conversation_info.items():
            if int(ukeys[1:]) in extracted_utterance_pos_list:
                extracted_conversation.append(utterance_info)

        # if len(extracted_conversation) != NUM_PREVIOUS_TURNS:
        #     print(len(extracted_conversation))

        return extracted_conversation



