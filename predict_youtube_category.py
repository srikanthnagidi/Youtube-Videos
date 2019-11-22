import pandas as pd

df = pd.read_json("merged_videoId_vs_Alt_id.txt", orient="index")

Category = ['Education', 'People & Blogs', 'Entertainment', 'Howto & Style', 'Science & Technology', 'Sports']

df = df[df.Category.isin(Category)]

data = df.loc[:,['Number of Dimensions citations',
       'Video mentions', 'cited_by_fbwalls_count', 'cited_by_feeds_count',
       'cited_by_gplus_count', 'cited_by_msm_count','cited_by_rdts_count', 'cited_by_tweeters_count',
       'cited_by_videos_count', 'cited_by_wikipedia_count', 'subno', 'dislikes', 'likes', 'views'] ]

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(20,10 ))
sns.heatmap(data.corr(), annot= True, ax=ax)

target = df.loc[:, "Category"]

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


#Random Forest Classifier--
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random.fit(traindata,traintarget.values.ravel())
trainresult = random.predict(traindata)
print(classification_report(traintarget,trainresult))
randomresult = random.predict(testdata)
print(classification_report(testtarget,randomresult))
sklearn.metrics.accuracy_score(testtarget, randomresult)

#Decision Tree Classifier
decision = DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes =10 , min_samples_leaf = 5, max_depth= 10)
decision.fit(traindata,traintarget)
train_result = decision.predict(traindata)
print(classification_report(traintarget,train_result))
decisionresult = decision.predict(testdata)
print(classification_report(testtarget,decisionresult))
sklearn.metrics.accuracy_score(testtarget, decisionresult)

#logistic regression
lr = LogisticRegression(random_state = 0)
lr.fit(traindata, traintarget.values.ravel())
regres = lr.predict(testdata)
print(classification_report(testtarget,regres))
sklearn.metrics.accuracy_score(testtarget, regres)