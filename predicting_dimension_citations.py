import pandas as pd
import ast
import numpy as np

df = pd.read_csv("Combined_with_scopus_subjects.csv", header=0)

df = df.fillna(0)
df = df [df.title != 0 ]
df = df[df.title != '[]']

def convert_si_to_number(x):
    total_stars = 0
    if 'K' in x:
        if len(x) > 1:
            total_stars = float(x.replace('K', '')) * 1000 # convert k to a thousand
    if 'M' in x:
        if len(x) > 1:
            total_stars = float(x.replace('M', '')) * 1000000 # convert M to a million
    return int(total_stars)

for index, row in df.iterrows():
    if (row['views'] =='0'):
        continue
    df.loc[index, "views"] = pd.to_numeric(ast.literal_eval(row['views']), errors='coerce').mean()
    df.loc[index, "likes"] = pd.to_numeric(ast.literal_eval(row['likes']), errors='coerce').mean()
    df.loc[index, "dislikes"] = pd.to_numeric(ast.literal_eval(row['dislikes']), errors='coerce').mean()
    df.loc[index, "CommentCount"] = pd.to_numeric(ast.literal_eval(row['CommentCount']), errors='coerce').mean()
    df.loc[index, "subno"] = np.mean([convert_si_to_number(number) for number in ast.literal_eval(row['subno'])])
    print(index)

#Topic modelling of abstracts.
df_abs = df[df.abstract != '0']

import re

from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 're', 'edu', 'use', 'new', 'may', 'could', 'say', 'get', 'go',
                   'know', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can', 'also', 'many', 
                   'do', 'be', 'also'])
stop_words = list(dict.fromkeys(stop_words))
len(stop_words)

import string
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def clean_data(summaries):
    # Remove Emails
    summaries = [re.sub('\S*@\S*\s?', '', sent) for sent in summaries]

    # Remove new line characters
    summaries = [re.sub('\s+', ' ', sent) for sent in summaries]

    # Remove distracting single quotes
    summaries = [re.sub("\'", "", sent) for sent in summaries]
    
    summaries = [strip_links(sent) for sent in summaries]

    summaries = [strip_all_entities(sent) for sent in summaries]
    
    summaries = [re.sub(r"(^|\W)\d+", "", sent) for sent in summaries]
    
    return summaries

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

summaries = df_abs.abstract.values.tolist()

summaries = clean_data(summaries)

summaries_to_words = list(sent_to_words(summaries))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(summaries_to_words, min_count=5, threshold=100) # higher threshold fewer phrases. 

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
summaries_words_nostops = remove_stopwords(summaries_to_words)

# Form Bigrams
summaries_words_bigrams = make_bigrams(summaries_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
summaries_lemmatized = lemmatization(summaries_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(summaries_lemmatized[0])

# Create Dictionary
id2word = corpora.Dictionary(summaries_lemmatized)

# Create Corpus
texts = summaries_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = models.LdaModel(corpus, num_topics=10, id2word=id2word, passes=4, alpha=[0.01]*10, eta=[0.01]*len(id2word.keys()))
pprint(lda_model.print_topics())

topics = [lda_model[corpus[i]] for i in range(len(summaries))]

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

document_topic = pd.concat([topics_document_to_dataframe(topics_document, 10) for topics_document in topics]).reset_index(drop=True).fillna(0)
  
document_topic.set_axis(["Abstract Topic "+ str(i) for i in range(10) ], axis='columns', inplace=True)
document_topic.head()

df_abs.index = pd.RangeIndex(len(df_abs.index))
df_merged = df_abs.merge(document_topic, left_index=True, right_index=True)


df_merged["Citations"] = (df_merged["Number of Dimensions citations"]>=26).astype(int)

data = df_merged.loc[:, ['cited_by_gplus_count', 'cited_by_wikipedia_count', 'cited_by_rdts_count',
       'Video mentions', 'cited_by_fbwalls_count','cited_by_tweeters_count','views', 'likes', 'dislikes', 'CommentCount', 'subno',
       'Abstract Topic 0', 
       'Abstract Topic 1', 'Abstract Topic 2',
       'Abstract Topic 3', 'Abstract Topic 4', 'Abstract Topic 5',
       'Abstract Topic 6', 'Abstract Topic 7', 'Abstract Topic 8',
       'Abstract Topic 9']]

data.fillna(0, inplace=True)
target =  df_merged.loc[:, 'Citations']

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

traindata,testdata,traintarget,testtarget = train_test_split(data, target, test_size=0.25)

#feature scaling
sc = StandardScaler()
traindata = sc.fit_transform(traindata)
testdata = sc.transform(testdata)

#Random Forest Classifier--
random = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
random.fit(traindata,traintarget.values.ravel())
randomresult = random.predict(testdata)
print(classification_report(testtarget,randomresult))
print(classification_report(traintarget,random.predict(traindata)))
sklearn.metrics.accuracy_score(testtarget, randomresult)


#logistic regression
lr = LogisticRegression(random_state = 0)
lr.fit(traindata, traintarget.values.ravel())
regres = lr.predict(testdata)
print(classification_report(testtarget,regres))
print(classification_report(traintarget,lr.predict(traindata)))
sklearn.metrics.accuracy_score(testtarget, regres)

#Decision Tree Classifier
decision = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decision.fit(traindata,traintarget)
decisionresult = decision.predict(testdata)
print(classification_report(testtarget,decisionresult))
sklearn.metrics.accuracy_score(testtarget, decisionresult)

