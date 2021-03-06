# -*- coding: utf-8 -*-
"""
Final project : complaints analysis

1. crawling - data load
2. data preprocessing
3. metadata analysis
4. modeling
5. evaluating
"""
from selenium import webdriver # 라이브러리에서 사용하는 모듈만 호출
import os
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import math
import re
import pandas as pd

'''prompt에서 설치 필요
conda activate tensorflow
pip install beautifulsoup4
pip install pandas
'''

pwd = os.getcwd() # 현재 경로 
print(pwd)
    

#def complain_crawler(name) :일단 각각 코실행드 한 후 사용자함수 처리하자!
# 1. dirver 경로/파일 지정 
driver = webdriver.Chrome("C:/Users/sdh04/OneDrive/바탕 화면/ITWILL-Final_project-main/chromedriver")

# 2. 민원 검색 url 
driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 민원기관 선택

# 3. 검색 입력상자 tag -> 검색조건 입력(중앙행정기관 내에서 세부기관 검색)
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[2]/a[2]/span').click()           # 상세 검색으로 늘려줍니다
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="searchInstType"]/option[3]'))) # 지자체 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="searchInstType"]/option[3]').click()              # 지자체 선택

wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="instListDiv"]/option[21]')))   # 경기도 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="instListDiv"]/option[21]').click()  # 경기도 선택


# 4. [검색] 버튼 클릭 ("//tag[@attr='value']/sub element")
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').send_keys(Keys.ENTER)
driver.implicitly_wait(3)   # 3초 대기(자원 loading)


# 5. Crawling
# 총 민원 건수
total_complain = driver.find_element_by_xpath('//*[@id="frm"]/div[2]/span/span').text 
total_complain = int(total_complain.replace(",",""))

print(total_complain)

# 페이지별 민원 개수
complain_per_page = 50

# 전체 페이지 수 계산
total_page = int(total_complain)/complain_per_page
total_page = math.ceil(total_page) # 페이지 소수불가 -> 올림처리


# 데이터를 담을 리스트
titles = []   # 민원 제목 모음
contents = [] # 민원 질문 모음
replies = []  # 민원 답변 모음

         
# 6. 웹 스크래핑 본문
driver.find_element_by_xpath('//*[@id="listCnt"]/option[5]').click() # 50개씩 보기 버튼 누르기

for i in range(total_page) : 
    if i < total_page-1 : # 첫 페이지(다음 페이지를 누르지 않고 긁어옵니다.)
        for i in range(complain_per_page):
            # 민원 제목 선택
            driver.find_element_by_xpath('//*[@id="frm"]/table/tbody/tr[%d]/td[2]/a'% (i+1)).click() 
            title = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_top').text
            titles.append(title)
            #내용이 있으면 담고
            try :
                content = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_c').text
                contents.append(content)
            #없으면 공백을 넣습니다.
            except :
                contents.append('')
            reply = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.ans > div').text
            replies.append(reply)
            driver.back()
        # 페이지 다음 버튼 누르는 코드 위치 변동을 통해 중간 elif문 제거하였습니다.    
        try : 
            driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click() # 다음 페이지로 넘어가는 버튼(11페이지)
        except :
            driver.find_element_by_css_selector('#frm > div.page_list > span.nep_p_next').click() #>>버튼이 없을 경우 >버튼을 누릅니다.

    elif i == total_page-1: # 마지막 페이지를 긁습니다.
       # driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click() #위에서 이미 >>버튼을 눌렀고, 마지막페이지라 없어도 되므로 삭제.
        for i in range(total_complain%50): # 전체 민원수 % 50으로 나머지 갯수만큼 for문이 반복됩니다.
            driver.find_element_by_xpath('//*[@id="frm"]/table/tbody/tr[%d]/td[2]/a'% (i+1)).click() 
            title = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_top').text
            titles.append(title)
            #내용이 있으면 담고
            try :
                content = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_c').text
                contents.append(content)
            #없으면 공백을 넣습니다.
            except :
                contents.append('')
            reply = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.ans > div').text
            replies.append(reply)
            driver.back()

driver.close()

len(titles)
print(titles)

len(contents)
print(contents)

len(replies)
print(replies)

#데이터 프레임 저장 및 csv파일화
dd = {'titles':titles, 'contents':contents, 'replies':replies}

dd.dtypes

my_data = pd.DataFrame(data=dd)


my_data.dtypes

my_data.to_csv('minwon_crawling4400.csv', index=False, encoding='utf-8-sig') 


df = pd.read_csv('D:/Naver MYBOX/Bigdata visualization course/ITWILL/minwon_crawling4400.csv', encoding='utf-8')


