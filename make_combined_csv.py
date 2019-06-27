# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:56:28 2019

@author: srika
"""

import os
import json
import pandas as pd

path = "C:/Users/srika/Youtube_dataSet"
files = os.listdir(path)
len(path)

for i in range(len(files)):
    files[i] = os.path.abspath(path + "/" + files[i])
files[0]

csv_file = open("combined_data.csv", 'w')

col = ['cited_by_fbwalls_count', 'cited_by_feeds_count', 'cited_by_gplus_count', 'cited_by_msm_count', 
       'cited_by_posts_count', 'cited_by_rdts_count', 'cited_by_tweeters_count', 'cited_by_videos_count']

stats = ['viewCount', 'likeCount',  'dislikeCount', 'favoriteCount','commentCount']
df = pd.DataFrame(columns = col+stats, dtype=object)

for i in range(len(files)):
    json_file = open(files[i], 'r')
    data = json.load(json_file)
    df.loc[i, 'altmetric_id'] = files[i][31:-4]
    for c in col:
        try:
            df.loc[i, c] = data[0][files[i][31:-4]][c]
        except KeyError:
            df.loc[i, c] = 0
    for stat in stats:
        stat_list = []
        for j in range(len(data[1]['youtube'])):
            try:
                stat_list.append(data[1]['youtube'][j]['items'][0]['statistics'][stat])
            except IndexError:
                stat_list.append(0)
            except KeyError:
                stat_list.append(0)
        df.at[i, stat] = stat_list
    json_file.close()
    
df.to_csv("combined_data.csv")
    