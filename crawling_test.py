#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_project

1. data crawling


"""

import urllib.request
import urllib.parse
import requests
from bs4 import BeautifulSoup

res = requests.get("http://google.com")
res.raise_for_status()
print('웹 연결 성공...')
#문제 생기면 여기서 바로 끝남! 쓸데없는 시간 낭비를 줄여줌



# 지은
# -*- coding: utf-8 -*-
"""
Final project : complaints analysis

1. crawling - data load
2. data preprocessing
3. metadata analysis
4. modeling
5. evaluating
"""

#찬영(2021.07.27 pm 18:00 추가)
from selenium import webdriver

#크롬 드라이버 load(각자 설치해놓으신 위치에 맞게!)
#driver = webdriver.Chrome("D:\\Naver MYBOX\\Bigdata visualization course\\ITWILL/5.Tensorflow/workspace/chap07_Face_detection/lecture00_web_crawling/chromedriver.exe")
driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 자동화 제어 사이트(크롤링 대상 사이트) open

#원하는 페이지를 여는 코드
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[2]/a[2]/span').click()#상세 검색으로 늘려줍니다
driver.find_element_by_xpath('//*[@id="searchInstType"]').click()
driver.find_element_by_xpath('')# 여기서 콤보 박스를 우리가 고를 수 있도록 코드를 짜보죠
driver.find_element_by_xpath('')
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').click#검색을 누릅니다

driver.close() #자동화 제어 사이트 종료


# selenum 사용
from selenium import webdriver # 라이브러리에서 사용하는 모듈만 호출
import os 

pwd = os.getcwd() # 현재 경로 
print(pwd)
    

def complain_crawler(name) : 
    # 1. dirver 경로/파일 지정
    # 각자 컴터에 맞게 수정!
    # driver = webdriver.Chrome("E:/ITWILL/5_Tensorflow/workspace/Final_chap/chromedriver_win32/chromedriver.exe") #
    
    # 2. 민원 검색 url 
    driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 민원을 검색할 수 있는 페이지
    
    # 3. 검색 입력상자 tag -> 검색조건 입력
    search_Nm = driver.find_element_by_name('pttnTypeNm')       # 전체, 민원유사사례, 민원질의응답, 정책질의응답
    search_Type = driver.find_element_by_name('searchInstType') # 중앙행정기관, 지방자치단체, 교육기관, 공공기관 
    search_Cd = driver.find_element_by_name('searchInstCd')     # 세부기관(element 이름 찾기)
 
    # 버튼 클릭
    driver.find_elements_by_xpath("//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]").click()

    # 데이터를 가져오기 위한 함수 
    driver.implicitly_wait(3)  # 3초 대기(자원 loading)
    mydata = driver.find_element_by class_name('tit') # class명이tit인 모든 것을 리스트로 가져와 mydata에 할당

    
