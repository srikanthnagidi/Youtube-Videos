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
import random
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


chrome = webdriver.Chrome('C:/ProgramData/chocolatey/bin/chromedriver.exe')

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

    for index, row in list(df.iterrows())[7433:7435]:
        try:
            proxy_index = random.randint(0, len(proxies) - 1)
            proxy = proxies[proxy_index] 
            req = Request('http://icanhazip.com')
            req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
            
            alt_id = row["Altmetric_id"]
            
            mylist=[]
            alt_req = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id))
            alt_rs= json.loads(alt_req.text)
            print(alt_req.text)
            mylist.append({str(alt_id):alt_rs})
            
            y_ids = row['youtube_ids'].split()
            y_list = []
            for y_id in y_ids:
                y_data = {}
                y_data['videoId'] = y_id
                chrome.get("https://www.youtube.com/watch?v=" + y_id)
                channel_id = chrome.find_element_by_xpath('//a[@class="yt-simple-endpoint style-scope yt-formatted-string"]')
                c_id = channel_id.get_attribute('href')
                y_data["channelId"] = c_id[32:]
                print("ID")
                v_title =chrome.find_element_by_xpath('//h1[@class="title style-scope ytd-video-primary-info-renderer"]')
                y_data["title"] = v_title.text
                print("title")
                v_description =  chrome.find_element_by_xpath('//div[@id = "description" and @class="style-scope ytd-video-secondary-info-renderer"]')
                y_data["description"] = v_description.text
                print("description")
                elements = chrome.find_elements_by_xpath(
                        '//yt-formatted-string[@class = "style-scope ytd-toggle-button-renderer style-text"]')
                for element in elements:
                    if "likes" in element.get_attribute("aria-label"):
                        likes = int(element.get_attribute("aria-label").replace(",", "").split()[0])
                        y_data['likeCount'] = likes
                    if "dislikes" in element.get_attribute("aria-label"):
                        dislikes = int(element.get_attribute("aria-label").replace(",", "").split()[0])
                        y_data["dislikeCount"] = dislikes
                print("likes and dislikes")
                v_count = chrome.find_element_by_xpath('//div[@id = "count" and @class="style-scope ytd-video-primary-info-renderer"]')
                count = int(v_count.text.replace(",", "").split()[0])
                y_data["viewCount"] = count
                print("viewCount")
                n_comments = chrome.find_element_by_xpath('//h2[@id="count" and @class="style-scope ytd-comments-header-renderer"]')
                n_comm = int(n_comments.text.replace(",", "").split()[0])
                y_data["commentCount"] = n_comm
                print("commentCount")
                y_list.append(y_data)
            mylist.append({"youtube":y_list})
            file = open(str(alt_id) + ".txt", "w+")
            json.dump(mylist, file)
            file.close()
        except NoSuchElementException:
            print('NoSuch')
            alt_id = row["Altmetric_id"]
            file = open(str(alt_id) + ".txt", "w+")
            file.close()
            pass
            
        except exceptions.StaleElementReferenceException:
            print('Stale')
            alt_id = row["Altmetric_id"]
            file = open(str(alt_id) + ".txt", "w+")
            file.close()
            pass
        
        if index % 5 == 0:
            getProx()
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]
            
    
        # Make the call
        try:
            my_ip = urlopen(req).read().decode('utf8')        
            print('#' + str(n) + ': ' + my_ip)
                
          
        except:  
            # If error, delete this proxy and find another one
            #global proxies
            del proxies[proxy_index]
            print('proxy deleted')
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]