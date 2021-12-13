# Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations

This is the implementation of our work on "Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations". This paper has been accepted to UMAP 2020. If you find our repository useful in your paper, please cite our paper.

### Paper

------

- Wanling Cai and Li Chen. 2020. Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations. In *Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization (UMAP '20)*, July 14-17, 2020. [Link](https://dl.acm.org/doi/abs/10.1145/3340631.3394856)

**Citation (Bibtex entry):**

```latex
@inproceedings{IARD,
  author = {Wanling Cai and Li Chen},
  title = {Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations},
  booktitle = {Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization}
  series = {UMAP '20},
  year = {2020},
} 
```



### Dataset

------

We used the IARD data (see below), which is also included in the data folder in this repo. 

> Intent Annotation of Recommendation Dialogue (IARD) Dataset [[Download](https://github.com/wanlingcai1997/umap_2020_IARD)]



### Code Dependency

****

**Python 3.7**

**Required Packages:** Scikit-learn, Scikit-multilearn, xgboost, TensorFlow, Keras, NLTK, gensim (for word-embedding)



### **Additional Required Resources**

------

- [Glove Embedding](https://nlp.stanford.edu/projects/glove/) - [Download](https://nlp.stanford.edu/data/glove.6B.zip)
- [Opinion Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) -  [Download](http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)



### Usage

Below are examples of how to run our codes for predicting user intents with ML models.

Go to the folder  ``\user_intent_prediction\machine_learning_model`` and run the example scripti. For instance:

```py
python Main.py \
--file_input_data ../../data/annotation_data.json \
--neural_model 0 \
--algorithm_adaption 0 \
--problem_transformation 1 \
--cross_validation 10 \
--feature_normalization 0 
--content_features 1 \
--discourse_features 1 \
--sentiment_features 1 \
--conversational_features 1 
--num_previous_turns 1 \
--problem_transformation_method BR \
--model_name XGBoost
```

Note: It would run a long time (more than 1 hour) if you follow this example, as we use the 5-fold cross validation to select the best hyper-parameter and evaluate the model with the 10-fold cross-validation.

### Code references

------

- [Scikit-multilearn](http://scikit.ml/userguide.html)
- [User Intent Prediction in Information-seeking Conversations](https://github.com/prdwb/UserIntentPrediction)