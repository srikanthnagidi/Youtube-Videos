# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:45:51 2019

@author: srika
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import ast
from datetime import datetime


df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

data = df.loc[:, ['cited_by_fbwalls_count', 'cited_by_gplus_count', 'cited_by_tweeters_count',
       'cited_by_feeds_count', 'cited_by_msm_count','cited_by_posts_count', 'cited_by_wikipedia_count', 
       'Video mentions', 'avg_like_dislike_ratio', 'Response_Rate', 'Facebook mentions','Twitter mentions',
       'News mentions', 'Blog mentions']]
target =  df.loc[:, 'Citations']

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

traindata,testdata,traintarget,testtarget = train_test_split(data, target, test_size=0.25)

#feature scaling
sc = StandardScaler()
traindata = sc.fit_transform(traindata)
testdata = sc.transform(testdata)

#Decision Tree Classifier
decision = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decision.fit(traindata,traintarget)
decisionresult = decision.predict(testdata)
print(classification_report(testtarget,decisionresult))
sklearn.metrics.accuracy_score(testtarget, decisionresult)

#Random Forest Classifier--
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random.fit(traindata,traintarget.values.ravel())
randomresult = random.predict(testdata)
print(classification_report(testtarget,randomresult))
sklearn.metrics.accuracy_score(testtarget, randomresult)

#Bernoulli Naive Bayes Classifier
nbayes = BernoulliNB()
nbayes.fit(traindata,traintarget.values.ravel())
nbayesresult = nbayes.predict(testdata)
print(classification_report(testtarget,nbayesresult))
sklearn.metrics.accuracy_score(testtarget, nbayesresult)

#logistic regression
lr = LogisticRegression(random_state = 0)
lr.fit(traindata, traintarget.values.ravel())
regres = lr.predict(testdata)
print(classification_report(testtarget,regres))
sklearn.metrics.accuracy_score(testtarget, regres)