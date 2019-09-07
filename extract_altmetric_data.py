# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:03:26 2019

@author: srika
"""

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests
import json

df_ids = pd.read_csv("altmetricIDS.csv", header=None)
df_ids.columns = ['Altmetrics_Id']

cols = ['altmetric_id', 'title', 'doi', 'tq',  'abstract', 'authors', 'publisher_subjects', 
        'cited_by_fbwalls_count', 'cited_by_feeds_count', 'cited_by_gplus_count', 'cited_by_msm_count', 
        'cited_by_posts_count', 'cited_by_rdts_count', 'cited_by_tweeters_count', 'cited_by_videos_count', 
        'cited_by_wikipedia_count', 'cited_by_accounts_count','subjects']

cookies = {'explorer_user': 'aGs3NkFPWmplUDVMSUtoaTlteTRUOENpZ1Rxb25lK215Zmg2VThwTHU3QVUrNlFpVExOb041Z1ZYTXNwTUJ1QS0tOFMrODJpdk9CZXp4b2FldWQ2V1JhZz09--ffde6776cfe966da417109e3ceac9544d87d20b4'}

df = pd.DataFrame(columns = cols, dtype=object)
def main():
    global df
    for index, rows in df_ids.loc[100000:, :].iterrows():
        index=index-100000
        alt_id = rows['Altmetrics_Id']
        try:
            req  = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id)).text
            req_js = json.loads(req)
        except:
            continue
        for col in cols:
            try:
                df.loc[index, col] = req_js[col]
            except KeyError:
                df.loc[index, col] = ""
        print(index, alt_id, " = done")

    df.to_csv("alt_data3.csv", index=False)
        
if __name__ == '__main__':
    main()