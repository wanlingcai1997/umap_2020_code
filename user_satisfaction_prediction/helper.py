import re
import json
from datetime import datetime
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize 
from nltk.stem import SnowballStemmer

def construct_feature_name_dict (feature_name_dict, feature_type, number_cur_features, current_features_name_list):
    start_feature_index = len(feature_name_dict)
    assert (number_cur_features == len(current_features_name_list))
    for i in range(number_cur_features):
        feature_index = start_feature_index + i
        feature_name = feature_type + "(" + str(current_features_name_list[i]) +")"
        feature_name_dict[feature_index]  = feature_name
    return feature_name_dict

def get_sorted_dict(dict):
    sorted_dict = {}
    tuple_list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    for t in tuple_list:
        sorted_dict[t[0]] = t[1]
    return sorted_dict

def store_result_json_file(file_name,data):
    path = 'result_analysis/'+file_name
    with open(path, 'w') as newfile:
        json.dump(data, newfile, indent=4)
    return 

def print_current_time():
    now = datetime.now()
    print("[Timestamp:", now, ']', end = '')

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = string.strip().lower()
    text = string.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    string = ' '.join(stemmed_words)
    return string

# def tokenizer(doc):
#     token_pattern = re.compile(r"(?u)\b\w\w+\b")
#     return token_pattern.findall(doc)


def remove_stopwords(string):
    STOPWORDS = set(stopwords.words('english'))

    word_tokens = word_tokenize(string) 
    filtered_text = [w for w in word_tokens if not w in STOPWORDS] 
    text = " " .join (filtered_text)
    return text

def load_sentiment_lexicon(lexicon_file):
    lexicon_dict = {}
    with open(lexicon_file) as fin:
        for line in fin:
            if line != '\n':
                line = line.strip()
                lexicon_dict[line] = 1

    return lexicon_dict


def clean_and_remove_stopwords(utterance_text): 
    utterance_text = [clean_str(text) for text in utterance_text]
    utterance_text = [remove_stopwords(text) for text in utterance_text]
    return utterance_text

def add_end_tokens(whole_utterance_dataframe):
    utterance_text = []
    for index, utterance_data in whole_utterance_dataframe.iterrows(): 
    
        role = utterance_data['role']
        current_utterance_text = utterance_data['text']
        if role == 'recommender':
            utterance_text.append(current_utterance_text + ' <Recommender-END>')
        if role == 'seeker':
            utterance_text.append(current_utterance_text + ' <User-END>')

    # print(utterance_text)
       
    return utterance_text

if __name__ == "__main__":
    string = '<fdfs-fsfs> asdad wdwna sda <USER-END>!'
    string_clean = cle(string)
    print(string_clean)