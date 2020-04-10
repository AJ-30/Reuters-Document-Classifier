import nltk
nltk.download('reuters')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

#Reuters ModApte Dataset
from nltk.corpus import reuters

print(" The reuters corpus has {} tags".format(len(reuters.categories())))
print(" The reuters corpus has {} documents".format(len(reuters.fileids())))

#Selecting documents with top 10 classes by frequency
len(reuters.fileids('earn'))

train_categories = [reuters.categories(i) for i in reuters.fileids() if i.startswith('training/')]
test_categories = [reuters.categories(i) for i in reuters.fileids() if i.startswith('test/')]

from os.path import expanduser
from collections import defaultdict
from nltk.corpus import reuters

home = expanduser("~")
id2cat = defaultdict(list)

top_cats = {cat:0 for cat in reuters.categories()}

for cat in reuters.categories():
  for cat_list in train_categories+test_categories:
    if cat in cat_list:
      top_cats[cat] += 1

import collections,itertools

sorted_cats = sorted(top_cats.items(), key=lambda kv: kv[1],reverse = True)
sorted_dict = collections.OrderedDict(sorted_cats)
ten_cats = collections.OrderedDict(itertools.islice(sorted_cats, 10))
ten_cats

def ten_cat_present(ide):
  for cat in list(ten_cats.keys()):
    if cat in reuters.categories(ide):
      return True
  return False

train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if (i.startswith('training/') and ten_cat_present(i))])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if (i.startswith('test/') and ten_cat_present(i))])

#Text Pre-processing
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
# The TfidfVectorizer needs input as list(strings) and not list(list(strings)). So, the output of tokenize is a string
from nltk.stem import PorterStemmer
import re # Regular Expressions library

def tokenize(text):
    """ function takes a text data and ouputs a list of strings after tokenizing"""
    text_alpha = re.sub(r'[^A-Za-z]',' ',text) # removing non-alphabet
    tokens = nltk.word_tokenize(text_alpha) # tokenizing the text
    stems = []
    stemmer = PorterStemmer()
    for item in tokens:
      if item not in stop_words: # removing the stop words
        stems.append(stemmer.stem(item)) # stemming the tokens
    return ' '.join(stems)

# This preprocesses the articles before sending them to the TfidfVectorizer. It may take around 20 seconds to run.
train_tokenized = [tokenize(article) for article in train_documents]
test_tokenized = [tokenize(article) for article in test_documents]

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
vectorizer = TfidfVectorizer(analyzer='word',max_df=0.9,min_df=int(3),sublinear_tf=True,stop_words='english')

#vectorised_train_documents = vectorizer.fit_transform(train_documents)
vectorised_train_documents = vectorizer.fit_transform(train_tokenized)
#vectorised_test_documents = vectorizer.transform(test_documents)
vectorised_test_documents = vectorizer.transform(test_tokenized)

print(" The shape of vectorised_train_documents is {} ".format(vectorised_train_documents.shape))
print(" The shape of vectorised_test_documents is {} ".format(vectorised_test_documents.shape))

import pandas as pd
train_data = pd.DataFrame(vectorised_train_documents.toarray())
train_data.columns = vectorizer.get_feature_names()
train_data.index.names = ['documents']
train_data.columns.names = ['features']
train_data.head()

test_data = pd.DataFrame(vectorised_test_documents.toarray())
test_data.columns = vectorizer.get_feature_names()
test_data.index.names = ['documents']
test_data.columns.names = ['features']
test_data.head()

# Label Encoding
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_categories)
test_labels = mlb.transform(test_categories)

# Classification ALgorithms
#Linear SVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

predictions = classifier.predict(test_data)

micro_f1 = f1_score(test_labels, predictions,average='micro')
macro_f1 = f1_score(test_labels, predictions,average='macro')
print(" The Microaveraged F1-score with LinearSVC model is {} ".format(micro_f1*100))
print(" The Macroaveraged F1-score with LinearSVC model is {} ".format(macro_f1*100))

