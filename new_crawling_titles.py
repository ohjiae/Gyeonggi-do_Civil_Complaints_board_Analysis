import urllib.request as req # 원격 서버 url 자료 요청 
from bs4 import BeautifulSoup # source -> html 파싱 

'''
url 쿼리를 이용한 새로운 titles 생성 코드입니다.
'''

# 1. url query 만들기 
base_url = "https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid"

res = req.urlopen(base_url+"?_csrf=1a8db2e5-e9c2-4d50-b7bc-d56fc8c19fb0&recordCountPerPage=20000&pageIndex=1&epUnionSn=&dutySctnNm=&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl=")
src = res.read()
data = src.decode('utf-8')

html = BeautifulSoup(data, 'html.parser')

a_tag = html.select('a[class="tit"]')

titles = []
for a in a_tag :
    cont = str(a.string)
    titles.append(cont.strip())
    
