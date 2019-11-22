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

df= df[df.link !='0']

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

vectorizer = TfidfVectorizer()
def cosine_sim(d1, d2):
    tfidf = vectorizer.fit_transform([d1, d2])
    return ((tfidf * tfidf.T).A)[0,1]

df_id_video["score"] = 0
for index, row in df_id_video.iterrows():
    df_id_video.loc[index, "score"] = cosine_sim(row['abstract'], row['transcript'])

df_links = pd.read_json("youtube_links.txt", orient='index')