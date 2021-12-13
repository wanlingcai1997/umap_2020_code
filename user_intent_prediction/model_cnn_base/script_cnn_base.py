# script.py

# Step1: preprocess_data.py

import os


file_input_data = '../../umap_dataset/data/whole_annotation_data_336_modified_taxonomy.json' 


content_features = 1 
structural_features = 1
sentiment_features = 1
conversational_features = 1

feature_normalization = 0

model_name = 'cnn_base'
max_sequence_length = 50
max_num_vocabulary = 20000
embedding_selection = 'glove'   # embed
embedding_dimension = [50, 100, 200] # embed_dim

convolutional_units = [128, 256, 512,1024] # conv_units
conv_stride_size = [2,3] # stride
conv_pooling_size = [2,3]  # pool
conv_dropout_rate = [0.0, 0.2, 0.4, 0.6] # dropout
dense_units = [128,256] # dense_units


cross_validation = 3

script = 'python Main.py '
script += '--file_input_data %s ' % file_input_data

script += '--content_features %s ' % content_features
script += '--structural_features %s ' % structural_features
script += '--sentiment_features %s ' % sentiment_features
script += '--conversational_features %s ' % conversational_features
script += '--feature_normalization %s ' % feature_normalization
script += '--model_name %s ' % model_name
script += '--cross_validation %s ' % cross_validation

# model configuration
model_config = '--max_sequence_length %d --max_num_vocabulary %d --embedding_selection %s ' % \
        (max_sequence_length,max_num_vocabulary, embedding_selection)

for embed_dim in embedding_dimension:
        model_config_embed_dim = '--embedding_dimension %d ' % embed_dim
        model_para_conv_unit = ''
        model_para_stride_pool = ''
        model_para_dropout = ''
        model_para_dense = ''
        for conv_units in convolutional_units:
                model_para_conv_unit = '--convolutional_units %d ' % conv_units
                for stride_size in conv_stride_size:
                        model_para_stride_pool = '--conv_stride_size %d --conv_pooling_size %d ' % (stride_size, stride_size)
                        for dropout_rate in conv_dropout_rate:
                                model_para_dropout = '--conv_dropout_rate %f ' % dropout_rate
                                for den_units in dense_units:
                                        model_para_dense = '--dense_units %d ' % den_units
                                
        

                                        result_script = '> result/%s_content_%d_structural_%d_sentiment_%d_conversational_%d_feat_normalization_%d_embed_%s_embed_dim_%d_conv_units_%d_stride_%d_pool_%d_dropout_%.1f_dense_units_%d.txt' \
                                                % (model_name, content_features,structural_features, sentiment_features,conversational_features, feature_normalization,embedding_selection,embed_dim, conv_units,stride_size, stride_size,dropout_rate, den_units) 
                                        result_script = '> result/%s_%d_%d_%d_%d_%d_embed_%s_%d_conv_units_%d_%d_%d_%.1f_dense_units_%d.txt' \
                                                % (model_name, content_features,structural_features, sentiment_features,conversational_features, feature_normalization,embedding_selection,embed_dim, conv_units,stride_size, stride_size,dropout_rate, den_units) 
                                        
                                        s = script + model_config + model_config_embed_dim + model_para_conv_unit + model_para_stride_pool + model_para_dropout+ model_para_dense + result_script
                                        print(s)
                                        # os.system(s)



	 







