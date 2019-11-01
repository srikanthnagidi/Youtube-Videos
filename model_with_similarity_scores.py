import pandas as pd
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

df_transcripts = pd.read_json("transcript1.txt", orient="index")

df_transcripts = df_transcripts.drop(["altmetric_id", "abstract"], axis=1)

df_transcripts = df_transcripts.dropna()
ids_with_transcripts = df_transcripts.id.values

df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

df_id_video = pd.DataFrame(columns = ["altmetric_id", "video_id", "abstract", "transcript"])

i=0
for index, row in df.iterrows():
    video_ids = ast.literal_eval(row['link'])
    for v_id in video_ids:
        if v_id in ids_with_transcripts:
            df_id_video.loc[i, "video_id"] = v_id
            df_id_video.loc[i, "altmetric_id"] = row['altmetric_id']
            df_id_video.loc[i, "abstract"] = row["abstract"]
    i=i+1
    print (index)
         
for index, row in df_id_video.iterrows():
    df_id_video.loc[index, "transcript"] = df_transcripts.loc[df_transcripts.id == row["video_id"], "transcript"].values[0]
    
df_id_video = df_id_video[df_id_video.abstract != "0"]

from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
def cosine_sim(d1, d2):
    tfidf = vectorizer.fit_transform([d1, d2])
    return ((tfidf * tfidf.T).A)[0,1]

df_id_video["score"] = 0
for index, row in df_id_video.iterrows():
    df_id_video.loc[index, "score"] = cosine_sim(row['abstract'], row['transcript'])
    

df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

cols = ['cited_by_gplus_count', 'cited_by_wikipedia_count', 'altmetric_id','number_of_days',
       'Video mentions', 'avg_like_dislike_ratio', 'Response_Rate', 'Facebook mentions','Twitter mentions', "subno"
       'News mentions', 'Blog mentions', 'views', 'likes', 'dislikes', 'CommentCount', 'Number of Dimensions citations']

df_id_video = pd.merge(df_id_video, df[cols], on="altmetric_id")

df_id_video["citation_class"] = (df_id_video["Number of Dimensions citations"] >= 55).astype(int)

for index, row in df_id_video.iterrows():
    if (row['views'] =='0'):
        continue
    df_id_video.loc[index, "views"] = pd.to_numeric(ast.literal_eval(row['views']), errors='coerce').mean()
    df_id_video.loc[index, "likes"] = pd.to_numeric(ast.literal_eval(row['likes']), errors='coerce').mean()
    df_id_video.loc[index, "dislikes"] = pd.to_numeric(ast.literal_eval(row['dislikes']), errors='coerce').mean()
    df_id_video.loc[index, "CommentCount"] = pd.to_numeric(ast.literal_eval(row['CommentCount']), errors='coerce').mean()
    print(index)
    
data = df_id_video.loc[:, ['cited_by_gplus_count', 'cited_by_wikipedia_count','number_of_days',
       'Video mentions', 'avg_like_dislike_ratio', 'Response_Rate', 'Facebook mentions','Twitter mentions',
       'News mentions', 'Blog mentions', 'views', 'likes', 'dislikes', 'CommentCount']]

target =  df_id_video.loc[:, 'citation_class']