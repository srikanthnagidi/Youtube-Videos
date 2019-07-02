# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:38:41 2019

@author: srika
"""

import requests
import pandas as pd
import json
import numpy as np
from bs4 import BeautifulSoup

from urllib.request import Request, urlopen
from fake_useragent import UserAgent

df = pd.read_csv("C://Users//srika//Youtube-Videos//youtube.csv")
df.columns = ["Altmetric_id", "youtube_ids"]

df.head()

ua = UserAgent() # From here we generate a random user agent
proxies = [] # Will contain proxies [ip, port]

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common import exceptions
from selenium.webdriver.support.ui import WebDriverWait 


chrome = webdriver.Chrome('D:/emotionaldata/chromedriver')

def getProx():
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')
         # Save proxies in the array
    global proxies
    proxies=[]
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
        'ip':   row.find_all('td')[0].string,
        'port': row.find_all('td')[1].string
      })
    #print(len(proxies))
    #print(proxies)
    
def random_proxy():
  return random.randint(0, len(proxies) - 1)

def main():
    
     # Retrieve latest proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')
    
    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
                'ip':   row.find_all('td')[0].string,
                'port': row.find_all('td')[1].string
                })
        print(len(proxies))
        print(proxies)


  # Choose a random proxy
    proxy_index = random_proxy()
    proxy = proxies[proxy_index] 
    
    global df
    
    for index, row in df.iterrows():
        getProx()
        proxy_index = random.randint(0, len(proxies) - 1)
        proxy = proxies[proxy_index] 
        req = Request('http://icanhazip.com')
        req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
        
        alt_id = row["Altmetric_id"]
        
        mylist[]
        alt_req = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id))
        alt_rs= json.loads(alt_req.text)
        mylist.append({str(alt_id):alt_rs})
        
        y_ids = row['youtube_ids'].split()
        for y_id in y_ids:
            you_req = chrome.get("https://www.youtube.com/watch?v=" + y_id)
            wait = WebDriverWait(chrome, 10)
            v_title = wait.until(EC.presence_of_element_located(
                           (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
            v_description =  wait.until(EC.presence_of_element_located(
                                         (By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
            elements = chrome.find_elements_by_xpath(
                    '//yt-formatted-string[@class = "style-scope ytd-toggle-button-renderer style-text"]')
            stats=[]
            for element in elements:
                if "likes" in element.get_attribute("aria-label"):
                    likes = int(element.get_attribute("aria-label").replace(",", "").split()[0])
                    stats.append({'likeCount':likes})
                if "dislikes" in element.get_attribute("aria-label"):
                    dislikes = int(element.get_attribute("aria-label").replace(",", "").split()[0])
                    stats.append({"dislikeCount":dislikes})
            v_count = wait.until(EC.presence_of_element_located(
                                         (By.CSS_SELECTOR,"div#count yt-view-count-renderer"))).text
            count = int(v_count.replace(",", "").split()[0])
            stats.append({"viewCount":count})
            
        