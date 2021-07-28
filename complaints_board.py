
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
                       


# 4. Crawling

# 염찬영 2021_07_28 17:00
# 페이지 넘기기 + 그 페이지에서 타이틀 txt를 따와서 리스트에 추가시키기.
complain_titles = []
for i in range(344):
    driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click()

    for i in range(10):
        titles = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a' % (i+1)).text
        complain_titles.append(titles)
    print(aa)    

    
   
# 지은 2021_07_28 17:00
# selenum 사용
from selenium import webdriver # 라이브러리에서 사용하는 모듈만 호출
import os
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import math
'''prompt에서 설치 필요
conda activate tensorflow
pip install beautifulsoup4
'''

pwd = os.getcwd() # 현재 경로 
print(pwd)
    

#def complain_crawler(name) :일단 각각 코실행드 한 후 사용자함수 처리하자!
# 1. dirver 경로/파일 지정 
driver = webdriver.Chrome("E:/ITWILL/5_Tensorflow/workspace/Final_chap/chromedriver_win32/chromedriver.exe") # 민원기관 

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


# 데이터를 가져오기 위한 함수 
#mydata_title = driver.find_element_by class_name('tit') # 제목 - class명이tit를 mydata리스트에 할당


# 총 민원 건수
total_complain = driver.find_element_by_xpath('//*[@id="frm"]/div[2]/span/span').text 
print(total_complain)

# 페이지별 민원 개수
complain_per_page = 10

# 전체 페이지 수 계산
total_page = int(total_complain)/complain_per_page
total_page = math.ceil(total_page) # 페이지 소수불가 -> 올림처리


# 5. 여러페이지 반복하기 
# 데이터를 담을 리스트
data_list = []

# 민원 제목과 내용 답변 수집 함수 정의
#def get_page_data():
    
# 민원제목 수집 - selector tr:nth-child(i) (i: for문)
title = driver.find_elements_by_css_selector('#frm > table > tbody > tr:nth-child(1) > td.left > a')

# 민원내용 수집 - xpath 일치
contents = driver.find_elements_by_xpath('//*[@id="txt"]/div[1]/div[2]/div/div[1]') 
                                       #  //*[@id="txt"]/div[1]/div[2]/div/div[1]/span[1]/span 
                                       # //*[@id="txt"]/div[1]/div[2]/div/div[1]
                                       # //*[@id="txt"]/div[1]/div[2]/div/div[1]
                                       # //*[@id="txt"]/div[1]/div[2]/div/div[1]
# 민원답변 수집 - selector일치
reply = driver.find_elements_by_css_selector('#txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top')
                                             #txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top > span:nth-child(1) > span
                                             #txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top
                                             #txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top    

# 세부항목이 '국세청'인 경우만 수집
for i in range(len(title)):
    data={}
    data['title'] = title[index].text
    data['contents'] = contents[index].text
    data['reply'] = reply[index].text
    print(data)
    data_list.append(data)
        

print('민원 자료 수집 시작')                                                 

# 첫 페이지 수집하고 시작
get_page_data()

# 버튼 눌러서 페이지를 이동해 가면서 계속 수집                                             
for page in range(1, total_page):
    try:
        print(str(page) + "page 수집 끝")
        # button_index = page % 10 + 2
        
        # 데이터 수집이 끝난 뒤 다음 페이지 버튼을 클릭
        driver.find_element_by_xpath()
        driver.implicitly_wait(3)
        
        # 10page 수집이 끝나서 11로 넘어가기 위해서는 >> 버튼 눌러야함
        if (page % 10 ==0):
            driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[5]/a/img')




driver.close()



# 함수 호출 : 민원 기관별 저장  
officelist = ["국세청"]

for i in officelist :
pwd = os.getcwd()      # 현재 경로 
os.mkdir(i)            # 현재 위치에 폴더 생성 
os.chdir(pwd+"/"+ i)   # 검색어 이용 하위폴더 생성 
get_page_data(i)       # get_page_data
os.chdir(pwd)          # 원래 위치 이동                                        
    

    
  
# 지애 (실패..) 2021_07_28 17:00
title_list = []
titles = driver.find_elements_by_css_selector('#frm > table')
for title in titles :
    print(title.text)
    title_list.append(titles.text)
    
    

    
# 서다현 2021-07-28 17:50
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
#driver = webdriver.Chrome("/Users/aegohc/ITWILL/final_project/chromedriver") # 각자 컴터에 맞게 수정!
driver = webdriver.Chrome("C:/Users/sdh04/OneDrive/바탕 화면/ITWILL-Final_project-main/chromedriver")
    
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

driver.find_element_by_css_selector('#listCnt > option:nth-child(5)').click()


# 4. Crawling
for i in range(69) : 
    titles = []
    
    driver.find_element_by_xpath('//*[@id="frm"]/div[3]/span[4]/a/img').click()

    for i in range(50) : 

        title = driver.find_element_by_css_selector('#frm > table > tbody > tr:nth-child(%d) > td.left > a'%(i+1)).text
        titles.append(title)
    
    #print(titles)
    
len(titles)

driver.close()