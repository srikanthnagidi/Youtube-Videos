# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:14:56 2019

@author: iamab
"""

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests
from time import sleep

df = pd.read_csv("altmetric_ids_2.csv", header=None)
df.columns = ['Altmetrics_Id']
df["youtube_links"] = ""
ids = np.array(df['Altmetrics_Id'])

cookies = {'explorer_user': 'QzlOYUpYbUd4RHRjNkd1NGpDa3d4UW1vbHVLZEs1bzB4U0c4ODlpckwyVExET0RCZ05DUURoblFpRW9RWXpaUC0tYWVydmtaSHJ3MjE3Rk9NeFZ4NzhnQT09--d90e55706b28b731cdcf13f119323cc64a252890'}

def get_youtube_ids(link):
    y_links = []
    r = requests.get(link, cookies=cookies).text
    bs = BeautifulSoup(r)
    possible_links = bs.find_all('a')
    for link in possible_links:
        if link.has_attr('href') and 'youtube' in link.attrs['href']:
            y_links.append(link.attrs['href'][32:])
    return y_links

def main():
    file = open('youtube_second.csv', 'w')
    for i in range(len(ids)):
        try:
            y_links = get_youtube_ids('https://www.altmetric.com/details/35956696/chapter/'+str(ids[i])+'/video')
            df.at[i, 'youtube_links'] = y_links
            file.write(str(ids[i]) + ",")
            file.write(" ".join(y_links) + "\n")
            if i%10 == 0:
                sleep(0.5)
            print(str(ids[i]) + "done")
        
        except:
            file.write(str(ids[i]) + "," + "\n")
    
    file.close()


if __name__ == '__main__':
  main()