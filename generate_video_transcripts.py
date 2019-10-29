import pandas as pd

df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)

df_abstracts = df.loc[:20000, ['altmetric_id', 'abstract', 'link']]

from langdetect import detect
from tqdm import tqdm_notebook
tqdm_notebook().pandas()

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

from youtube_transcript_api import YouTubeTranscriptApi

import ast
df_id_alt = pd.DataFrame(columns = ['id', 'transcript'])
v_ids=set()
for index, row in df_abstracts.iterrows():
    v_ids = v_ids.union(set(ast.literal_eval(row['link'])))

for v_id in v_ids:
    df_id_alt.loc[i, 'id'] = v_id
    try:
        d = YouTubeTranscriptApi.get_transcript(v_id, languages=['en'])
        st = ""
        for lt in d:
            st = st + " " + lt['text']
        df_id_alt.loc[i, 'transcript'] = st
        i=i+1
    except YouTubeTranscriptApi.CouldNotRetrieveTranscript:
        i=i+1
        continue
    print(v_id)
    
df_id_alt.to_csv("transcript1.csv", index=False)
df_id_alt.to_json("transcript1.txt", orient = "index")