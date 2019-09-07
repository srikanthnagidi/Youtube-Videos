# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:23:17 2019

@author: srika
"""

import pandas as pd

df_alt = pd.read_csv("alt_data.csv", header=0)
df_you = pd.read_csv("youtube_all_data_cleaned.csv", header =0)
df_link = pd.read_csv("youtube.csv", header = 0)

y_cols =['views','likes','dislikes', "CommentCount"]
alt_cols = ['altmetric_id','cited_by_fbwalls_count', 'cited_by_gplus_count', 'cited_by_tweeters_count', 'cited_by_rdts_count',
            'cited_by_videos_count', 'Number of Dimensions citations'] 
cols = y_cols + alt_cols

df = pd.DataFrame(columns = cols, dtype=object)

for index, row in df_alt.iterrows():
    alt_id = row['altmetric_id']
    for c in alt_cols:
        df.loc[index, c] = df_alt[index, c]
    y_ids = df_link.loc[df_link['altmetric_ids'] == alt_id, 'youtube_ids']
    y_ids = y_ids.split()
    for cl in y_cols:
        s =0
        for y_id in y_ids:
            s = s + df_you.loc[df_you['link'] == y_id, cl]
        avg_s = s/len(y_ids)
        df.loc[index, cl] = avg_s
        
                