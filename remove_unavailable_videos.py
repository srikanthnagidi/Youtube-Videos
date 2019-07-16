# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:20:34 2019

@author: srika
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException
from selenium.webdriver.support import expected_conditions as EC

df = pd.read_csv("unique_ids.csv", header=0)
df.head()

driver = webdriver.Chrome("C:/ProgramData/chocolatey/bin/chromedriver.exe")
#driver.maximize_window()
driver.get("https://www.youtube.com/")
#wait = WebDriverWait(driver, 10)

links = ["dqw4w9wgxcq", "9nPTDYqGcdE"]
df["status"] = ""
for y_id in df.youtube_ids.values[:30]:
    driver.get("https://www.youtube.com/watch?v=" + y_id)
    driver.implicitly_wait(30)
    avail_text = driver.find_element_by_xpath('//ytd-watch-flexy').text
    if "unavailable" in avail_text:
        df.status[df.youtube_ids == y_id] = "U"
        print(y_id,  " unavailable")
    else:
        df.status[df.youtube_ids == y_id] = "A"
        print(y_id,  " available")
driver.close()


    