from sklearn.feature_extraction.text import TfidfVectorizer
import re
import copy
from nltk import word_tokenize 
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import spacy
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helper import load_sentiment_lexicon
import numpy as np
import helper
import sys

def initialize_fitted_vectorizer (whole_utterances):
    whole_utterances_text = whole_utterances[['text']].values  

    whole_utterances_text = [helper.clean_str(text[0]) for text in whole_utterances_text]
    whole_utterances_text = [helper.remove_stopwords(text) for text in whole_utterances_text]
    vectorizer = TfidfVectorizer(min_df = 2, stop_words='english', ngram_range=(1,2))
   
    fitted_vectorizer = vectorizer.fit(whole_utterances_text)
    return fitted_vectorizer


# ------------------------------------------------------------#
# -------  content features ----------------------------------#
# ------------------------------------------------------------#

def content_tf_idf(utterances, fitted_vectorizer):
    # Text preprocessing 
    
    utterances_text = [helper.clean_str(text[0]) for text in utterances]
    utterances_text = [helper.remove_stopwords(text) for text in utterances_text]
    
    tf_idf_utterance_vector = fitted_vectorizer.transform(utterances_text).toarray()

    return tf_idf_utterance_vector


def get_occurred_entity_label_set(utterances):
    nlp = spacy.load("en_core_web_sm")
    # collect all entity_label_set(change code position)
    entity_label_set = []
    for utterance in utterances:
        utterance_doc = nlp(utterance)
        for ent in utterance_doc.ents:  
            if ent.label_ not in entity_label_set:
                entity_label_set.append(ent.label_) 
    helper.print_current_time()
    print("entity_label_set (%d):" % len(entity_label_set), end='')
    print(entity_label_set)
    return  entity_label_set

def content_name_entity(utterances, entity_label_set):
    feat = []
    nlp = spacy.load("en_core_web_sm")

    for utterance in utterances:
        utterance_doc = nlp(utterance)
        ent_frequency = {}
        utterance_ent_feat = []
        if len(utterance_doc.ents) > 0: 
            for ent in utterance_doc.ents:
                if ent.label_ in ent_frequency:
                    ent_num = ent_frequency[ent.label_] + 1
                    ent_frequency[ent.label_] = ent_num
                else:
                    ent_frequency[ent.label_] = 1
            for each_ent in entity_label_set:
                if each_ent in ent_frequency:
                    utterance_ent_feat.append(ent_frequency[each_ent])
                else:
                    utterance_ent_feat.append(0)
        else:
            utterance_ent_feat = [0 for i in range(len(entity_label_set))]

        feat.append(copy.deepcopy(utterance_ent_feat))
    # print("len(feat):",len(feat))
    # print("len(feat[0]):",len(feat[0]))
    return feat

def content_num_movies(utterances):
    feat = []
    for utterance in utterances:
        movie_id_list = re.findall(r'@[0-9]+', str(utterance))
        # if (len(movie_id_list)):
        #     for movie_id in movie_id_list:
        #         print(movie_id)
        
        feat.append(len(movie_id_list))
    return feat

# def content_semantic_embedding(utterances):
#     feat = []
#     nlp = spacy.load("en_core_web_sm")
#     for utterance in utterances:
#         utterance_doc = nlp(utterance)
#         doc_vector = utterance_doc.vector
    
#         feat.append(doc_vector)
#     print("content_semantic_embedding")
#     print("len(feat):",len(feat))
#     print("len(feat[0]):",len(feat[0]))
#     return feat

def extract_content_features(utterances, fitted_vectorizer, entity_label_set):
    utterances_text = utterances[['text']].values.reshape(-1).tolist()
    content_features = [[] for i in range(len(utterances))]

    f_content_tf_idf = content_tf_idf(utterances[['text']].values, fitted_vectorizer)
    # entity_label_set = get_occurred_entity_label_set(utterances_text)
    f_content_name_entity = content_name_entity(utterances_text, entity_label_set)
    f_content_num_movies = content_num_movies(utterances_text)
    content_features = np.column_stack((f_content_tf_idf, f_content_name_entity, f_content_num_movies))
    
    # check feature vector
    # check_feature_vector(f_content_tf_idf, 'f_content_tf_idf')
    # check_feature_vector(f_content_name_entity, 'f_content_name_entity')
    # check_feature_vector(f_content_num_movies, 'f_content_num_movies')
    # check_feature_vector(content_features, 'content_features')


    return content_features

# ------------------------------------------------------------#
# --------  Discourse features -------------------------------#
# ------------------------------------------------------------#

def get_occurred_POS_tag_set(utterances):
    nlp = spacy.load("en_core_web_sm")
    # collect all universal POS tag list (change code position)
    POS_tag_set = []
    for utterance in utterances:
        utterance_doc = nlp(utterance)
        for token in utterance_doc:
            if token.pos_ not in POS_tag_set:
                POS_tag_set.append(token.pos_)
    helper.print_current_time()
    print("POS_tag_set (%d):" % len(POS_tag_set), end='')
    print(POS_tag_set)
    return POS_tag_set

def discourse_POS_features(utterances, POS_tag_set):
    feat = []

    nlp = spacy.load("en_core_web_sm")
    
    for utterance in utterances:
        utterance_doc = nlp(utterance)
        pos_frequency = {}
        utterance_pos_feat = []
        for token in utterance_doc:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            #     token.shape_, token.is_alpha, token.is_stop)
            if token.pos_ in pos_frequency:
                pos_num = pos_frequency[token.pos_] + 1
                pos_frequency[token.pos_] = pos_num
            else:
                pos_frequency[token.pos_] = 1
        for each_pos in POS_tag_set:
            if each_pos in pos_frequency:
                utterance_pos_feat.append(pos_frequency[each_pos] / len(utterance_doc))
            else:
                utterance_pos_feat.append(0)
        feat.append(copy.deepcopy(utterance_pos_feat))
        
    # print("len(feat):",len(feat))
    # print("len(feat[0]):",len(feat[0]))
    return feat



def discourse_5W1H(utterances):
    feat = []
    # Original taken from https://github.com/prdwb/UserIntentPrediction/blob/master/features/features/content_features.py

    for utterance in utterances:
        wh_vector = [0] * 6
        
        wh_vector[0] = 1 if utterance.find('how') != -1 else 0
        wh_vector[1] = 1 if utterance.find('what') != -1 else 0
        wh_vector[2] = 1 if utterance.find('why') != -1 else 0
        wh_vector[3] = 1 if utterance.find('who') != -1 else 0
        wh_vector[4] = 1 if utterance.find('where') != -1 else 0
        wh_vector[5] = 1 if utterance.find('when') != -1 else 0

        feat.append(list(map(str, wh_vector)))
    return feat 


def discourse_question_mark(utterances):
    feat = []
    for utterance in utterances:
        if '?' in utterance:
            feat.append(1)
        else:
            feat.append(0)
    return feat


def discourse_exclamation_mark(utterances):
    feat = []
    for utterance in utterances:
        if utterance.find('!') != -1:
            feat.append(1)
        else:
            feat.append(0)
    return feat
 

     
def discourse_utterance_length (utterances):
    utterance_length = []
    utterance_unique_length = []
    utterance_unique_stemmed_length = []

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for utterance in utterances:
        word_tokens = word_tokenize(utterance) 
        words = [w for w in word_tokens if not w in stop_words] 

        unique_words = list(set(words))
        unique_words_stemmed = list(set([stemmer.stem(w) for w in unique_words]))

        utterance_length.append(len(words))
        utterance_unique_length.append(len(unique_words))
        utterance_unique_stemmed_length.append(len(unique_words_stemmed))


    return utterance_length, utterance_unique_length, utterance_unique_stemmed_length 


def extract_discourse_features(utterances, POS_tag_set):
    utterances_text = utterances[['text']].values.reshape(-1).tolist()
    discourse_features = [[] for i in range(len(utterances))]

    # POS_tag_set = get_occurred_POS_tag_set(utterances_text)
    f_discourse_POS_features = discourse_POS_features(utterances_text, POS_tag_set)
    f_discourse_5W1H = discourse_5W1H(utterances_text)
    f_discourse_question_mark = discourse_question_mark(utterances_text)
    f_discourse_exclamation_mark = discourse_exclamation_mark(utterances_text)

    f_discourse_length, f_discourse_unique_length, f_discourse_unique_stem_length = discourse_utterance_length(utterances_text)
        

    discourse_features = np.column_stack((f_discourse_POS_features, f_discourse_5W1H, f_discourse_question_mark, f_discourse_exclamation_mark,\
         f_discourse_length, f_discourse_unique_length, f_discourse_unique_stem_length))


    # check feature vector
    # check_feature_vector(f_discourse_POS_features, 'f_discourse_POS_features')
    # check_feature_vector(f_discourse_5W1H, 'f_discourse_5W1H')
    # check_feature_vector(f_discourse_question_mark, 'f_discourse_question_mark')
    # check_feature_vector(f_discourse_exclamation_mark, 'f_discourse_exclamation_mark')
    # check_feature_vector(f_discourse_length, 'f_discourse_length')
    # check_feature_vector(f_discourse_unique_length, 'f_discourse_unique_length')
    # check_feature_vector(f_discourse_unique_stem_length, 'f_discourse_unique_stem_length')
    # check_feature_vector(discourse_features, 'discourse_features')

    return discourse_features


# ------------------------------------------------------------#
# -------  sentiment features --------------------------------#
# ------------------------------------------------------------#
def sentiment_thanks (utterances):
    feat = []
    for utterance in utterances:
        if utterance.find('thank') != -1:
            feat.append(1)
        else:
            feat.append(0)
    return feat


def sentiment_scores(utterances):
    feat = []
    sentiment_analyzer = SentimentIntensityAnalyzer()
    for utterance in utterances:
        ss = sentiment_analyzer.polarity_scores(utterance)
        feat.append(list(map(str, [ss['neg'], ss['neu'], ss['pos']])))
    return feat

def sentiment_opinion_lexicon(utterances, pos_file, neg_file):
    feat = []

    pos_dict = load_sentiment_lexicon(pos_file)
    neg_dict = load_sentiment_lexicon(neg_file)

    for utterance in utterances:
        pos_count, neg_count = 0, 0
        word_tokens = word_tokenize(utterance) 

        for word in word_tokens:
            if word in pos_dict:
                pos_count += 1
            elif word in neg_dict:
                neg_count += 1
        feat.append([str(pos_count), str(neg_count)])
    return feat




def extract_sentiment_features(utterances, pos_file, neg_file):

    utterances_text = utterances[['text']].values.reshape(-1).tolist()
    sentiment_features = [[] for i in range(len(utterances))]
 
    f_sentiment_thanks = sentiment_thanks(utterances_text)
    f_sentiment_scores = sentiment_scores(utterances_text)
    f_sentiment_opinion_lexicon = sentiment_opinion_lexicon(utterances_text, pos_file, neg_file)
    sentiment_features = np.column_stack((f_sentiment_thanks,  f_sentiment_scores, f_sentiment_opinion_lexicon))
    
    # check feature vector
    # check_feature_vector(f_sentiment_thanks, 'f_sentiment_thanks')
    # check_feature_vector(f_sentiment_scores, 'f_sentiment_scores')
    # check_feature_vector(f_sentiment_opinion_lexicon, 'f_sentiment_opinion_lexicon')
    # check_feature_vector(sentiment_features, 'sentiment_features')

    return sentiment_features 


def check_feature_vector(feature_vector, featur_name):
    helper.print_current_time()
    num_features = 1 if type(feature_vector[0]) == int else len(feature_vector[0])
    print(" %s - Shape: (%d, %d)" % (featur_name, len(feature_vector),num_features))
    

def extract_features(utterances, fitted_vectorizer,  entity_label_set, POS_tag_set, content_flag, discourse_flag, sentiment_flag, pos_file, neg_file):

    # utterances_text = utterances[['text']].values.reshape(-1).tolist()
    content_features = [[] for i in range(len(utterances))]
    discourse_features = [[] for i in range(len(utterances))]
    sentiment_features = [[] for i in range(len(utterances))]
    

    if content_flag == 1:
        content_features = extract_content_features(utterances, fitted_vectorizer, entity_label_set)
        
    if discourse_flag == 1:
        discourse_features = extract_discourse_features(utterances, POS_tag_set)

       
    if sentiment_flag == 1:
        sentiment_features = extract_sentiment_features(utterances, pos_file, neg_file)

       
    feature_vector = np.array(np.column_stack((content_features, discourse_features, sentiment_features))).astype('float64')
    
    # check feature vector
    # check_feature_vector(feature_vector, 'feature_vector')
    
    return feature_vector 



if __name__ == "__main__":
    utterances = ["play, played, plays? played. Hi!, my name is Aaron. What's yours"]
    print(sentiment_scores(utterances))