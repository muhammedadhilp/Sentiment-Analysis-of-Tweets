# Sentiment Analysis of Tweets
## Introduction
This project involves sentiment analysis of a large dataset of tweets. The objective is to classify tweets into positive and negative sentiments using natural language processing (NLP) and machine learning techniques. The dataset used in this project contains tweets labeled with sentiment scores, and we employ text preprocessing and classification algorithms to build a predictive model.

## Dataset Description
- Source: The dataset is sourced from kaggle url:https://www.kaggle.com/datasets/kazanova/sentiment140 and contains a total of 1.6 million tweets.
- Columns:
- sentiment: Sentiment score (0 for negative, 4 for positive).
- id: Unique tweet identifier.
- date: Date of the tweet.
- query: Query related to the tweet.
- user: Twitter username.
- tweet: The content of the tweet.
- 
## Exploratory Data Analysis (EDA)
Data Sampling: Due to the large size of the dataset, a sample of 20,000 tweets is used for analysis.

## Data Distribution:

The dataset is well-balanced between positive and negative sentiments, as shown by the count plot of sentiments.
```python

sns.countplot(x='sentiment', data=df)
```
## Sentiment Distribution:

Positive tweets : 9872
Negative tweets : 10128
Total number of tweets:20000

## Data Preprocessing
Column Removal: Unnecessary columns are removed to focus on sentiment and tweet content.

```python

df = df.drop(['id', 'date', 'query', 'user'], axis=1)
```
Sentiment Mapping: Sentiment scores are converted from 4 to 1 to simplify the classification into binary positive (1) and negative (0).

```python

df['sentiment'] = df['sentiment'].replace(4, 1)
```
Text Cleaning: Tweets are processed to remove URLs, usernames, specific words, punctuation, and stopwords. Contractions and abbreviations are expanded.

```python

def process_tweets(tweet):
    ...
```
Tokenization: Tweets are tokenized, and words are lemmatized and filtered based on length.

Abbreviations: Common abbreviations are expanded for better understanding.

```python

def abbreviate(tweet):
    ...
```
Final Text Cleaning: Words with fewer than 3 letters are removed to ensure meaningful analysis.

## Model Training and Evaluation
Text Vectorization: The cleaned tweets are vectorized using CountVectorizer.

```python

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(df['processed_tweet'].values.astype('U'))
```
Train-Test Split: The dataset is split into training and testing sets.

```python

X_train, X_test, y_train, y_test = train_test_split(text_counts, y, test_size=0.20, random_state=19)
```
Model Training: The ComplementNB model is trained and evaluated.

```python

cnb = ComplementNB()
cnb.fit(X_train, y_train)
```
Cross-Validation: Cross-validation scores are calculated to evaluate the model's performance.

```python

cross_cnb = cross_val_score(cnb, X, y, n_jobs=-1)
```
Accuracy:

Training Accuracy: {train_acc_cnb:.2f}%
Testing Accuracy: {test_acc_cnb:.2f}%
Plotting Results: A bar chart is plotted to visualize the training and testing accuracy.

python
Copy code
data_cnb = [train_acc_cnb, test_acc_cnb]
labels = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(data_cnb)), labels)
plt.ylabel('Accuracy')
plt.title('Accuracy Plot with Best Parameters')
plt.bar(range(len(data_cnb)), data_cnb, color=['blue', 'darkorange'])
Conclusion
The sentiment analysis model successfully classifies tweets into positive and negative sentiments with an accuracy of {train_acc_cnb:.2f}% on the training set and {test_acc_cnb:.2f}% on the testing set. The preprocessing steps, including text cleaning and abbreviation expansion, significantly contribute to the model's performance. Future work may include experimenting with different models and hyperparameters to further enhance accuracy.

Requirements
pandas
numpy
matplotlib
seaborn
nltk
sklearn
re
string
pickle
Ensure that all required packages are installed, and run the provided script to preprocess the data, train the model, and evaluate its performance.
