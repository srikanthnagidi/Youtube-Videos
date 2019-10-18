
import pandas as pd
import ast

df_ = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

df_abstract = pd.read_csv("alt_abstracts_topics.csv", header=0)

df = df_[df_.abstract!='0']
df = pd.merge(df, df_abstract[['altmetric_id', 'Topics']], on = "altmetric_id")


for index, row in df.iterrows():
    if (row['views'] =='0'):
        continue
    df.loc[index, "views"] = pd.to_numeric(ast.literal_eval(row['views']), errors='coerce').mean()
    df.loc[index, "likes"] = pd.to_numeric(ast.literal_eval(row['likes']), errors='coerce').mean()
    df.loc[index, "dislikes"] = pd.to_numeric(ast.literal_eval(row['dislikes']), errors='coerce').mean()
    df.loc[index, "CommentCount"] = pd.to_numeric(ast.literal_eval(row['CommentCount']), errors='coerce').mean()
    df.at[index, "subno"]= ast.literal_eval(row["subno"])
    print(index)
    
df.subno = (df.subno.replace(r'[KM]+$', '', regex=True).astype(float) * 
            df.subno.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))

df = df.fillna(0)
data = df.loc[:, ['cited_by_gplus_count', 'cited_by_wikipedia_count', 
       'Video mentions', 'avg_like_dislike_ratio', 'Response_Rate', 'Facebook mentions','Twitter mentions',
       'News mentions', 'Blog mentions', 'views', 'likes', 'dislikes', 'CommentCount', 'Topics']]

df["Citations"] = (df["Number of Dimensions citations"]>=26).astype(int)

target =  df.loc[:, 'Citations']

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(20,10 ))
sns.heatmap(data.corr(), annot= True, ax=ax)

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
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
random.fit(traindata,traintarget.values.ravel())
randomresult = random.predict(testdata)
print(classification_report(testtarget,randomresult))
sklearn.metrics.accuracy_score(testtarget, randomresult)
