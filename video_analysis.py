import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import ast
from datetime import datetime


df = pd.read_csv("combined_all_data.csv", header=0)
df['number_of_days'] = None
for index, row in df.iterrows():
    try:
        st = ast.literal_eval(row["pubdate"])
    except ValueError:
        continue
    enddate = datetime.strptime("Jul 30, 2019", "%b %d, %Y")
    mylist = []
    for item in st:
        if len(item) == 1 or item == 'Series([], )':
            mylist.append(0)
        else:
            try:
                published_date = datetime.strptime(item[-12:].lstrip(), "%b %d, %Y")
                mylist.append((enddate - published_date).days)
            except ValueError:
                mylist.append(0)
    df.at[index, 'number_of_days'] = mylist
    print (index)
df['avg_like_dislike_ratio'] = None
for index, rows in df.iterrows():
    ratio =[]
    try:
        likes = pd.to_numeric(ast.literal_eval(rows['likes']))
    except ValueError:
        continue
    dislikes = pd.to_numeric(ast.literal_eval(rows['dislikes']))
    for i in range(len(likes)):
        if dislikes[i] ==0:
            ratio.append(likes[i])
        else:
            ratio.append(likes[i]/dislikes[i])
    df.loc[index, 'avg_like_dislike_ratio'] = np.mean(ratio)
    print(index)

df['Response_Rate'] =None
for index, rows in df.iterrows():
    ratio = []
    try:
        viewlist = pd.to_numeric(ast.literal_eval(rows['views']))
    except ValueError:
        continue
    likes = pd.to_numeric(ast.literal_eval(rows['likes']))
    dislikes = pd.to_numeric(ast.literal_eval(rows['dislikes']))
    for i in range(len(viewlist)):
        if (viewlist[i] ==0):
            ratio.append(0)
        else:
            ratio.append((likes[i]+dislikes[i])*100/viewlist[i])
    df.loc[index, 'Response_Rate'] = np.mean(ratio)
    print(index)

df = df.fillna(0)
df_news = pd.read_csv("news_and_blogs.csv", header=0)

df_merge = pd.merge(df, df_news[['Facebook mentions','Twitter mentions', 'altmetric_id', 'News mentions', 'Blog mentions']],
                    on = "altmetric_id", how='left')
data = df_merge.loc[:, ['cited_by_fbwalls_count', 'cited_by_gplus_count', 'cited_by_tweeters_count',
       'cited_by_feeds_count', 'cited_by_msm_count','cited_by_posts_count', 'cited_by_wikipedia_count', 
       'Video mentions', 'avg_like_dislike_ratio', 'Response_Rate', 'Facebook mentions','Twitter mentions',
       'News mentions', 'Blog mentions']]
target =  df_merge.loc[:, 'Citations']
df_merge.to_csv("combined_with_news_n_blogs.csv", index=False)

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

#Random Forest Classifier
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