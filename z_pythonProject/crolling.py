from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.keys import Keys
import csv

# Crome 에 대한 최신 버젼을 가져와서 웹 페이지를 가져와서 랜더링을 한다.
chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

driver.get(url='https://www.youtube.com/watch?v=29eBJsJesKA')

# 페이지 스크롤링
for i in range(0, 300):
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
    time.sleep(1)  # 스크롤 사이에 대기 시간 추가

titles = driver.find_elements(By.CSS_SELECTOR, '#content-text')
print(titles)

# CSV 파일로 저장
csv_file_path = '/Users/kimkyusan/Desktop/김규산_국비지원/test/pythonProject/youtube_reply_16.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['title'])

    for title in titles:
        print(title.text)
        writer.writerow([title.text])

driver.quit()
