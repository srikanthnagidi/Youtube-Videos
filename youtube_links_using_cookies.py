# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:38:17 2019

@author: iamab
"""

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests
from time import sleep

df = pd.read_csv("missing_youtube_ids.csv", header=None)
df.columns = ['Altmetrics_Id']
df["youtube_links"] = ""
ids = np.array(df['Altmetrics_Id'])

cookies = {'explorer_user': 'anZ0c3lMMkx3VklDRUt1RkMyR3hmdmVZZlE3ZXB0ZlprYW84WkQraCtBc08vaXpDUEY4WFYzUWdlN2hsS1FnNy0taVZrNlQyOXduaGw5bHNxTGVIN21jdz09--52145f21346de5d6f994772fc975ce0050d15136'}


def get_all_pages(main_link):
    myset= set()
    r = requests.get(main_link, cookies=cookies).text
    bs = BeautifulSoup(r)
    possible_links = bs.find_all('a')
    for link in possible_links:
        if link.has_attr('href') and "video/page:" in link.attrs['href']:
            myset.add(link.attrs['href'])
    return myset

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
    file = open('youtube_missing.csv', 'w')
    for i in range(len(ids)):
        try:
            y_links = get_youtube_ids('https://www.altmetric.com/details/' + str(ids[i]) + '/video')
            myset = get_all_pages('https://www.altmetric.com/details/' + str(ids[i]) + '/video')
            for link in myset:
                y_links.extend(get_youtube_ids(link))
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
