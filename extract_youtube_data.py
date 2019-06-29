# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:55:01 2019

@author: srika
"""

import pandas as pd
import requests
import json
import numpy as np

df = pd.read_csv("Part1.csv", header=None)

df.columns = ['Altmetrics_Id', "youtube_ids"]

youtube_param = {"key" : ""}

youtube_api_url = "https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics"

ids = np.array(df['Altmetrics_Id'])

def main():
    for alt_id in ids:
        try:
            mylist = []
            alt_req = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id))
            alt_rs= json.loads(alt_req.text)
            mylist.append({str(alt_id):alt_rs})
            youtube_ids = df[df["Altmetrics_Id"]==alt_id]['youtube_ids'].to_string()
            y_ids = youtube_ids.split()[1:]
            youtube_data = []
            for you_id in y_ids:
                youtube_param["id"] = you_id
                response_statistics = requests.get(youtube_api_url, youtube_param)
                if (response_statistics.status_code != 200):
                    break
                json_statistics = json.loads(response_statistics.text)
                youtube_data.append(json_statistics)
            mylist.append({'youtube':youtube_data})
            file = open(str(alt_id)+".txt", "w+")
            json.dump(mylist, file)
            file.close()
            print(str(alt_id) + " done")
            
        except:
            print(str(alt_id) + " not done")
            
if __name__ == '__main__':
  main()
