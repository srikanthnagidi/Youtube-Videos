import requests
import json
import numpy as np
from bs4 import BeautifulSoup
from lxml import html
import pandas as pd
from urllib.request import Request, urlopen
import random

from fake_useragent import UserAgent

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common import exceptions

from time import sleep

df =pd.read_csv("altmetricIDS.csv", header=None)
df.columns = ['Altmetrics_Id']
df["youtube_links"] = ""

ids = np.array(df['Altmetrics_Id'])

chrome = webdriver.Chrome('C:/ProgramData/chocolatey/bin/chromedriver.exe')

ua = UserAgent() # From here we generate a random user agent
proxies = [] # Will contain proxies [ip, port]

# Retrieve latest proxies
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
    
def get_all_pages(main_link):
    myset= set()
    chrome.get(main_link)
    elements = chrome.find_elements_by_xpath("//a[@href]")
    for elem in elements:
        if "video/page:" in elem.get_attribute("href"):
            myset.add(elem.get_attribute("href"))
    return myset

def get_youtube_ids(link):
    y_links = []
    chrome.get(link)
    elements = chrome.find_elements_by_xpath("//a[@href]")
    for elem in elements:
        if "youtube" in elem.get_attribute("href"):
            y_links.append(elem.get_attribute("href")[32:])
    return y_links

def make_csv(ids): 
    for i in range(len(ids)):
        # Every 2 requests, generate a new proxy
        if i%2 == 0:
            getProx()
            random_index = random.randint(0, len(proxies) - 1)
            proxy = proxies[random_index]
        # Make the call
        
        try:
            req = Request('http://icanhazip.com')
            req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
            sleep(0.5)
            y_links = get_youtube_ids('https://www.altmetric.com/details/' + str(ids[i]) + '/video')
            myset = get_all_pages('https://www.altmetric.com/details/' + str(ids[i]) + '/video')
            for link in myset:
                y_links.extend(get_youtube_ids(link))
            df.at[i, 'youtube_links'] = y_links

        except:
            pass
        
        try:
            my_ip = urlopen(req).read().decode('utf8')        
            print('#' + str(i) + ': ' + my_ip)

        except:  
            # If error, delete this proxy and find another one
            #global proxies
            del proxies[random_index]
            print('proxy deleted')
            random_index = random.randint(0, len(proxies) - 1)
            proxy = proxies[random_index]