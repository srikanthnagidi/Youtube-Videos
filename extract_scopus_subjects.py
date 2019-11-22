import pandas as pd
import ast
import json
import requests


df = pd.read_csv("combined_all_data.csv", header=0, dtype=object)
df["scopus_subjects"] = [[]] * len(df) 
for index , row in df.loc[79871:, :].iterrows():
    alt_id = row['altmetric_id']
    response = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id))
    try:
         df.loc[index, 'scopus_subjects'] = dict(json.loads(response.content))['scopus_subjects']
    except:
        df.loc[index, 'scopus_subjects'] = [] 
    print( index, df.loc[index, 'scopus_subjects'])
    

df.to_csv("Combined_with_scopus_subjects.csv", index=False)
