# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:52:46 2019

@author: iamab
"""

import pandas as pd

df = pd.read_csv("C://Users//iamab//Desktop//Data youtube//youtube.csv")
df.columns = ["Altmetric_id", "youtube_ids"]

df.head()
df_ids = pd.DataFrame(columns = ["youtube_ids"], dtype=str)

for index, row in df.iterrows():
    if row['youtube_ids'] == None:
        continue
    try:
        y_ids = row['youtube_ids'].split()
        print(index , row["Altmetric_id"])
        for y_id in y_ids:
            df_ids.loc[len(df_ids)] = y_id
    except AttributeError:
        print(index , row["Altmetric_id"], " not done")
        continue

df_ids = df_ids.drop_duplicates()
len(df_ids)

df_ids.to_csv("unique_ids.csv", index=False)