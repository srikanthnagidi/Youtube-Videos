# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:57:13 2019

@author: iamab
"""
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException
from selenium.webdriver.support import expected_conditions as EC

df_id = pd.read_csv("C://Users//iamab//Desktop//Data youtube//video_ids_and_status.csv", header=0)
df_id.head()

links=["https://www.youtube.com/watch?v=R59sfLVdBJA","https://www.youtube.com/watch?v=YFSwJuJqekw&list=RDR59sfLVdBJA&index=3"]
#links from dataset
df = pd.DataFrame(columns = ['link', 'title','views','likes','dislikes','subname','subno','pubdate','description', "Category", "CommentCount"])

driver = webdriver.Chrome("C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe")
driver.maximize_window()
driver.get("https://www.youtube.com/")
wait = WebDriverWait(driver, 10)
for index, row in df_id.iterrows():
    x=row["youtube_ids"]
    if (row["status"] == 'U'):
        df.loc[len(df)] = [x, "", "","","","","","","","", ""]
        print (str(len(df)-1), x , " = unavailable")
        continue
    try:
       driver.get("https://www.youtube.com/watch?v=" + x)
       driver.implicitly_wait(10)
       v_test = driver.find_element_by_css_selector("yt-formatted-string.more-button.style-scope.ytd-video-secondary-info-renderer")
       v_test.click()
       driver.execute_script("window.scrollTo(0, 3000);")
       v_id = x
       v_title = wait.until(EC.presence_of_element_located((By.XPATH,"//h1[@class='title style-scope ytd-video-primary-info-renderer']"))).text
       v_views = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"yt-view-count-renderer span.view-count.style-scope.yt-view-count-renderer"))).text
       v_likes = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"yt-formatted-string#text.style-scope.ytd-toggle-button-renderer.style-text"))).get_attribute("aria-label")
       v_pub = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"span.date.style-scope.ytd-video-secondary-info-renderer"))).text
       v_desc = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"yt-formatted-string.content.style-scope.ytd-video-secondary-info-renderer"))).text
       v_dislikes = driver.find_elements_by_xpath('//yt-formatted-string[@id= "text" and @class = "style-scope ytd-toggle-button-renderer style-text"]')[1].get_attribute("aria-label")
       try:
           v_subno = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"span.deemphasize.style-scope.yt-formatted-string"))).text
       except:
           v_subno = 0
       v_subname = driver.find_elements_by_css_selector("a.yt-simple-endpoint.style-scope.yt-formatted-string")[0].text
       #driver.execute_script("window.scrollTo(0, 3000);")
       driver.implicitly_wait(10)
       try:
           v_comments = driver.find_element_by_xpath('//yt-formatted-string[@class="count-text style-scope ytd-comments-header-renderer"]').text
       except:
           v_comments = 0
       v_cat = driver.find_element_by_xpath('//div[@id="content" and @class = "style-scope ytd-metadata-row-renderer"]').text
       df.loc[len(df)] = [v_id, v_title, v_views, v_likes, v_dislikes, v_subname, v_subno, v_pub, v_desc,  v_cat, v_comments]
       
       print (str(len(df)-1), x , " = done")

    except NoSuchElementException:
        df.loc[len(df)] = [x, "", "","","","","","","","", ""]
        print (str(len(df)-1), x , " = not done")

    except TimeoutException:
        df.loc[len(df)] = [x, "", "","","","","","","","", ""]
        print (str(len(df)-1), x , " = not done")

    except ElementNotInteractableException:
        df.loc[len(df)] = [x, "", "","","","","","","","",""]
        print (str(len(df)-1), x , " =not done")


df.to_csv("youtube_all_data.csv", index= False)
driver.close()

