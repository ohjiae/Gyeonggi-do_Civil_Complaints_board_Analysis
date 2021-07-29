  
    
    
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


'''prompt에서 설치 필요
conda activate tensorflow
pip install beautifulsoup4
'''

pwd = os.getcwd() # 현재 경로 
print(pwd)
    

#def complain_crawler(name) :일단 각각 코실행드 한 후 사용자함수 처리하자!
# 1. dirver 경로/파일 지정 
driver = webdriver.Chrome("/Users/aegohc/ITWILL/final_project/chromedriver")

# 2. 민원 검색 url 
driver.get("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid") # 민원기관 선택

# 3. 검색 입력상자 tag -> 검색조건 입력(중앙행정기관 내에서 세부기관 검색)
#search_Nm = driver.find_element_by_name('pttnTypeNm')       # 전체, 민원유사사례, 민원질의응답, 정책질의응답
#search_Type = driver.find_element_by_name('searchInstType') # 중앙행정기관, 지방자치단체, 교육기관, 공공기관 
search_Cd = driver.find_element_by_name('searchInstCd')      # 세부기관(element 이름 찾기)


driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[2]/a[2]/span').click()           # 상세 검색으로 늘려줍니다
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="searchInstType"]/option[2]'))) # 중앙행정기관 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="searchInstType"]/option[2]').click()              # 중앙행정기관 선택

wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="instListDiv"]/option[34]')))   # 국세청 선택 가능할때까지 기다리기
driver.find_element_by_xpath('//*[@id="instListDiv"]/option[34]').click()                # 국세청 선택


# 4. [검색] 버튼 클릭 ("//tag[@attr='value']/sub element")
driver.find_element_by_xpath('//*[@id="frm"]/div[1]/div[1]/div[4]/button[1]').send_keys(Keys.ENTER)
driver.implicitly_wait(3)   # 3초 대기(자원 loading)


# 총 민원 건수
total_complain = driver.find_element_by_xpath('//*[@id="frm"]/div[2]/span/span').text 
total_complain = total_complain.replace(",","")

print(total_complain)

# 페이지별 민원 개수
complain_per_page = 10

# 전체 페이지 수 계산
total_page = int(total_complain)/complain_per_page
total_page = math.ceil(total_page) # 페이지 소수불가 -> 올림처리

# 5. 여러페이지 반복하기 
# 데이터를 담을 리스트

titles = [] # 민원 제목 모음
contents = [] # 민원 질문 모음
replies = [] # 민원 답변 모음

# 민원 제목 선택
for i in range(69) :
    driver.find_element_by_xpath('//*[@id="frm"]/table/tbody/tr[%d]/td[2]/a'% (i+1)).click() 
    title = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_top').text
    titles.append(title)
    content = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.mw > div > div.samC_c').text
    contents.append(content)
    reply = driver.find_element_by_css_selector('#txt > div.same_mwWrap > div.samBox.ans > div').text
    replies.append(reply)
    driver.back()









