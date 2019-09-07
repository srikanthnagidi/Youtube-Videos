from selenium import webdriver
import pandas as pd
import numpy as np

df = pd.read_csv("unique_ids_2.csv", header=0 )
#https://www.youtube.com/watch?v=55JMv3Y_u0o
#links from dataset
links=["https://www.youtube.com/watch?v=55JMv3Y_u0o","https://www.youtube.com/watch?v=YFSwJuJqekw&list=RDR59sfLVdBJA&index=3"]

driver = webdriver.Chrome("C:/ProgramData/chocolatey/bin/chromedriver.exe")
driver.get("https://www.youtube.com/")

df["status"] = ""
for y_id in np.array(df.iloc[:, 0]):
    driver.get("https://www.youtube.com/watch?v=" + y_id)
    driver.implicitly_wait(30)
    avail_text = driver.find_element_by_xpath('//ytd-watch-flexy').text
    if "unavailable" in avail_text:
        df.status[df.youtube_ids == y_id] = "U"
        print(y_id,  " unavailable")
    else:
        df.status[df.youtube_ids == y_id] = "A"
        print(y_id,  " available")

df.to_csv("video_ids_and_status_3.csv", index= False)
driver.close()