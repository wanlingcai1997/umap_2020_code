# script.py

# Step1: preprocess_data.py

import os

def execute_script (script, feature_script, model_script, result_script):
    s = script + feature_script +  model_script # + result_script
    print(s)
    # input()
    os.system(s)

    return s

file_input_data = '../../umap_dataset/data/whole_annotation_data_336_modified_taxonomy.json' 

num_previous_turns = 1

# model_name = ['LR','LinearSVC', 'MLP','NB_M', 'KNN', 'DT', 'RF', 'AdaBoost','GBDT','XGBoost']
# Tree
# model_name = ['DT', 'RF', 'AdaBoost','GBDT','XGBoost']

# model_name = ['LR','LinearSVC', 'MLP','DT', 'RF', 'AdaBoost','GBDT','XGBoost', 'NB_M','NB_G', 'SVM', 'KNN']
model_name = ['KNN']
content_features_choice = [0]
discourse_features_choice = [0]
sentiment_features_choice = [1]
conversational_features_choice = [0]
task_specific_features_choice = [0]
# content_features_choice = [1,0]
# discourse_features_choice = [1,0]
# sentiment_features_choice = [1,0]
# conversational_features_choice = [1,0]
# task_specific_features_choice = [0]

feature_normalization = 0
cross_validation = 10

script_list = []

script = 'python Main.py '
script += '--file_input_data %s ' % file_input_data
script += '--num_previous_turns %d ' % num_previous_turns
script += '--feature_normalization %s ' % feature_normalization
script += '--cross_validation %s ' % cross_validation


for content_features in content_features_choice:
    content_features_s = '--content_features %s ' % content_features
    for discourse_features in discourse_features_choice:
        discourse_features_s = '--discourse_features %s ' % discourse_features
        for sentiment_features in sentiment_features_choice:
            sentiment_features_s = '--sentiment_features %s ' % sentiment_features
            for conversational_features in conversational_features_choice:
                conversational_features_s = '--conversational_features %s ' % conversational_features
                for task_specific_features in task_specific_features_choice:
                    task_specific_features_s = '--task_specific_features %s ' % task_specific_features
                    if content_features + discourse_features + sentiment_features + conversational_features + task_specific_features == 0:
                        continue
                    feature_script = content_features_s + discourse_features_s + sentiment_features_s + conversational_features_s + task_specific_features_s
         

                    for model in model_name:
                        model_script = '--model_name %s ' % model
                        result_script = '> result/%s_pre_turns_%d_content_%d_discourse_%d_sentiment_%d_conversation_%d_task_%d_feat_normalization_%d_cv_%d.txt' % \
                            (model,num_previous_turns, content_features,discourse_features, sentiment_features, conversational_features, task_specific_features, feature_normalization, cross_validation) 
                        script_list.append( execute_script (script, feature_script,  model_script, result_script)) 


	 

print("Total script: %d pieces." % len(script_list))





