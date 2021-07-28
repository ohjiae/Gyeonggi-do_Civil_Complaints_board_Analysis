
"""
Final project : complaints analysis

1. crawling - data load
2. data preprocessing
3. metadata analysis
4. modeling
5. evaluating
"""
# 모듈 import (라이브러리에서 사용하는 모듈만 호출)
# selenum 사용
from selenium import webdriver
import os 
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By  # 기다리는 조건 설정 시 탐색용 
from selenium.webdriver.support.ui import WebDriverWait  # 기다리는 용도 (비동기)
from selenium.webdriver.support import expected_conditions as EC  # 기다릴 때의 조건 설정용
from selenium.webdriver.common.keys import Keys  # 키 입력용


# 경로 확인
pwd = os.getcwd() # 현재 경로 
print(pwd)



#def complain_crawler(name) : 
# 1. dirver 경로/파일 지정 
driver = webdriver.Chrome("/Users/aegohc/ITWILL/final_project/chromedriver") # 각자 컴터에 맞게 수정!
    
# 2. 민원 검색 url 
driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 민원을 검색할 수 있는 페이지

    
# 3. div Element 가져오기

current_total_pages = driver.find_element_by_xpath('//*[@id="frm"]/div[2]/span/span').text #현재 총 검색결과 수 
print(current_total_pages)


driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[2]/a[2]/span').click()#상세 검색으로 늘려줍니다
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="searchInstType"]/option[2]'))) #중앙행정기관 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="searchInstType"]/option[2]').click() #중앙행정기관 선택

wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="instListDiv"]/option[34]'))) #국세청 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="instListDiv"]/option[34]').click() #국세청 선택


driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').send_keys(Keys.ENTER) #검색 클릭이 안되서 엔터로 대신
                       
#push_srch_bt = driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').submit # 검색을 누릅니다


# 4. Crawling
# 페이지 넘기기 + 그 페이지에서 타이틀 txt를 따와서 리스트에 추가시키기. 염찬영 2021_07_28 17:00
aa = []
for i in range(344):
    driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click()

    for i in range(10):
        a = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a' % (i+1)).text
        aa.append(a)
    print(aa)    
