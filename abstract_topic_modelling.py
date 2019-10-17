import pandas as pd
from pprint import pprint
df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

import re
import numpy as np

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


from langdetect import detect
from tqdm import tqdm_notebook
tqdm_notebook().pandas()

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
    
    summaries = list(filter(None, summaries))
    
    return summaries

df_abstracts = df.loc[:, ['altmetric_id', 'abstract']]

df_abstracts=df_abstracts.dropna() 
df_abstracts['lang'] = ""

for index, row in df_abstracts.iterrows():
    try:
        lg= detect(row['abstract'])
    except:
        lg = ""
    df_abstracts.loc[index, 'lang'] = lg
    print(index)

df_abstracts = df_abstracts[df_abstracts.lang == 'en']

descriptions = df_abstracts.abstract.values.tolist()

descriptions = clean_data(descriptions)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
descriptions_to_words = list(sent_to_words(descriptions))

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Remove Stop Words
descriptions_words_nostops = remove_stopwords(descriptions_to_words)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(descriptions_words_nostops, min_count=5, threshold=100) # higher threshold fewer phrases. 

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Bigrams
descriptions_words_bigrams = make_bigrams(descriptions_words_nostops)


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Do lemmatization keeping only noun, adj, vb, adv
descriptions_lemmatized = lemmatization(descriptions_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(descriptions_lemmatized[0])

# Create Dictionary
desc_id2word = corpora.Dictionary(descriptions_lemmatized)

# Create Corpus
texts = descriptions_lemmatized

# Term Document Frequency
desc_corpus = [desc_id2word.doc2bow(text) for text in texts]

import os
from gensim.models.wrappers import LdaMallet
os.environ['MALLET_HOME'] = 'C:/Users/srika/mallet-2.0.8/'
mallet_path = "C:/Users/srika/mallet-2.0.8/bin/mallet"
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=desc_corpus, num_topics=10, id2word=desc_id2word)
pprint(ldamallet.print_topics())


def format_topics_sentences(ldamodel, corpus, texts=descriptions):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
        print(i)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamallet, desc_corpus, texts=df_abstracts.abstract.values.tolist())


df_dominant_topic = df_topic_sents_keywords.reset_index()
# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
dominant_topics = df_dominant_topic.Dominant_Topic.values.tolist()

df_abstracts['Topics'] = pd.Series(dominant_topics, index = df_abstracts.index)
df_abstracts.loc[:, "Topics"].value_counts()