from selenium import webdriver # 라이브러리에서 사용하는 모듈만 호출
import os
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import math


#def complain_crawler(name) :일단 각각 코실행드 한 후 사용자함수 처리하자!
# 1. dirver 경로/파일 지정 
#driver = webdriver.Chrome("E:/ITWILL/5_Tensorflow/workspace/Final_chap/chromedriver_win32/chromedriver.exe") 
driver = webdriver.Chrome("D:\\Naver MYBOX\\Bigdata visualization course\\ITWILL/5.Tensorflow/workspace/chap07_Face_detection/lecture00_web_crawling/chromedriver.exe")

# 2. 민원 검색 url 
driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 신문고 민원사례 게시판

# 3. 검색 입력상자 tag -> 검색조건 입력(중앙행정기관 내에서 세부기관 검색)
# search_Nm = driver.find_element_by_name('pttnTypeNm')     # 전체, 민원유사사례, 민원질의응답, 정책질의응답
# search_Type = driver.find_element_by_name('searchInstType') # 중앙행정기관, 지방자치단체, 교육기관, 공공기관 
# search_Cd = driver.find_element_by_name('searchInstCd')     # 세부기관(element 이름 찾기)

# print(search_Cd, search_Type, search_Nm)


driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[2]/a[2]/span').click()           # 상세 검색으로 늘려줍니다
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="searchInstType"]/option[2]'))) # 중앙행정기관 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="searchInstType"]/option[2]').click()              # 중앙행정기관 선택

wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="instListDiv"]/option[34]')))   # 국세청 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="instListDiv"]/option[34]').click()                # 국세청 선택

# wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="instListDiv"]/option[68]'))) # 해양경찰청
# driver.find_element_by_xpath('//*[@id="instListDiv"]/option[68]').click()  

# 4. [검색] 버튼 클릭 ("//tag[@attr='value']/sub element")
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').send_keys(Keys.ENTER)
driver.implicitly_wait(3)   # 3초 대기(자원 loading)

# 5. 정렬 갯수를 10개 -> 50개로 변경
driver.find_element_by_css_selector('#listCnt > option:nth-child(5)').click()


# 6. Crawling test



# 6-1. crawling titles(test)

# 총 민원 건수 (변수명 current_total_pages -> total_complain)
import re
total_complain = driver.find_element_by_xpath('//*[@id="frm"]/div[2]/span/span').text 
def comma_num_to_int(text):
  n = re.sub(",", "", text)
  return int(n)
total_complain = comma_num_to_int(total_complain)

print(total_complain)

# 페이지별 민원 개수
complain_per_page = 50

# 전체 페이지 수 계산 
total_page = int(total_complain)/complain_per_page
total_page = math.ceil(total_page) # 페이지 소수불가 -> 올림처리
print(total_page)

titles = [] # 제목을 저장하는 list 공간 선언

for i in range(total_page) : 
    if i == 0: # 첫 페이지(다음 페이지를 누르지 않고 긁어옵니다.)
        for i in range(50):
            title = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a'%(i+1)).text
            titles.append(title)
    elif i < total_page-1: # 2~마지막 전 페이지(다음 페이지를 누르고, 50개를 긁어옵니다)
        driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click()
        for i in range(50):
            title = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a'%(i+1)).text
            titles.append(title)              
    elif i == total_page-1: # 마지막 페이지를 긁습니다.
        driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click()
        for i in range(total_complain%50): # 전체 민원수 % 50으로 나머지 갯수만큼 for문이 반복됩니다.
            title = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a'%(i+1)).text
            titles.append(title)              
         
print(titles)    

