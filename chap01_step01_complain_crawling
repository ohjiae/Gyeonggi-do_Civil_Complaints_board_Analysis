import urllib.request as req # 원격 서버 url 자료 요청 
from bs4 import BeautifulSoup # source -> html 파싱 
import re # 정규식 활성화 패키지
from tqdm import tqdm # for문 진행상황 확인 가능한 외부 package

'''
url 쿼리를 이용한 웹사이트 scraping 코드입니다.
csrf토큰은 사용자마다, ip마다, 날짜마다 달라지므로 수정할 필요가 있습니다. token 추출 함수를 따로 만들고 싶었는데 실력도 시간도 모자라네요.
'''

#1.url 생성
base_url = "https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid"

#2. url read & decode to utf-8
res = req.urlopen(base_url+"?_csrf=eea0ef35-d28a-4e60-ad98-e92df36b6643&recordCountPerPage=20000&pageIndex=1&epUnionSn=&dutySctnNm=&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl=")
src = res.read()
data = src.decode('utf-8')

# 3. 정규식 표현으로 각 게시글의 고유 주소 넘버 추출(17000개 한번에!)
p = re.compile(r'\d\w\w-\d\d\d\d-\d\d\d\d\d\d\d-\d\w\w-\d\d\d\d-\d\d\d\d\d\d\d-\d\d\d\d\d\d\d-\d\d') 
#. = 하나의 문자 / ^ : 문자열의 시작
# $ (se$) : 문자열의 끝 -> case, base (o), face (x)
address = p.findall(data)

base_url2 = "https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseDetail.npaid"

# 4. 고유 주소 넘버를 for문으로 돌면서 제목, 답변을 긁어오는 코드
titles = []
# contents = [] --> 현재 경기도 data에는 거의 답변이 없는 것으로 확인되어 우선 삭제 후 보기로 결정
replies = []


for i in tqdm(address):
    try:
        res = req.urlopen(base_url2+"?_csrf=eea0ef35-d28a-4e60-ad98-e92df36b6643&recordCountPerPage=1&pageIndex=1&epUnionSn=%s&dutySctnNm=taol&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl="% i)
        src = res.read()
        data = src.decode('utf-8')
        html = BeautifulSoup(data, 'html.parser')
        tit = html.select('#txt > div.same_mwWrap > div.samBox.mw > div > div > strong') # 제목 추출
        ans = html.select('#txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top') # 답변 추출
        titles.append(str(tit[0].string)) # 저장한 제목을 string 형태로 추출하여 titles 리스트에 저장
        replies.append(str(ans[0].text).strip()) # 저장한 답변에서 text data만 따로 추출, 공백을 제거(strip)한 이후 리스트에 저장
    except:
        pass # 혹시 문제가 생긴다면 넘어갈 수 있도록 설계. 별로 좋은 방법은 아닙니다.(에러를 specific하게 규정하는게 좋음)

# 5. save to csv
import pandas as pd

dic = {'title':titles, 'replies':replies}

df = pd.DataFrame(dic)

df.to_csv('crawlingdata17326.csv', encoding='utf-8-sig')


##아래는 강사님이 주신 코드인데, 필요한거만 빼서 코드에 추가했습니다. 혹 참고하실 분은 참고하세요~

# for rows in answers:  
#     #print(rows)
#     #print(type(rows))
#     '''
#     <class 'str'> - None
#     <class 'list'> - [<div class="samC_top">...]
#     '''    
#     if rows != 'None' : # None 제외
#         row = rows[0] # <div> 태그 선택 
#         print(rows[0].text) # <div> 태그 내용 추출 
#         text = str(rows[0].text).strip() # 불용어(공백) 제거 
#         # 문장 마지막 날짜(2021-07-07) 제거 -> 불용어 제거 -> list 추가 
#         answers_text.append(text[:-10].strip()) 

# print(answers_text)
