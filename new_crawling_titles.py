import urllib.request as req # 원격 서버 url 자료 요청 
from bs4 import BeautifulSoup # source -> html 파싱 

'''
url 쿼리를 이용한 새로운 titles 생성 코드입니다.
'''

#1.url 생성
base_url = "https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid"

#2. url read & decode to utf-8
res = req.urlopen(base_url+"?_csrf=1a8db2e5-e9c2-4d50-b7bc-d56fc8c19fb0&recordCountPerPage=20000&pageIndex=1&epUnionSn=&dutySctnNm=&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl=")
src = res.read()
data = src.decode('utf-8')

# #3. parsing & variable save
# html = BeautifulSoup(data, 'html.parser')

# a_tag = html.select('필요 코드 주소')

# 3. 정규식 표현으로 각 게시글의 고유 주소 넘버 추출(17000개 한번에!)
p = re.compile(r'\d\w\w-\d\d\d\d-\d\d\d\d\d\d\d-\d\w\w-\d\d\d\d-\d\d\d\d\d\d\d-\d\d\d\d\d\d\d-\d\d') 
#. = 하나의 문자 / ^ : 문자열의 시작
# $ (se$) : 문자열의 끝 -> case, base (o), face (x)
address = p.findall(soup)

add = address[3000:3010]

for i in add:
    res = ("%s"% i)
    print(res)

for i in address:
    try:
        res = req.urlopen(base_url+"?_csrf=c196acf4-7186-412f-8c94-2a2e95d0fd6a&recordCountPerPage=1&pageIndex=1&epUnionSn=%s&dutySctnNm=taol&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl="% i)
        src = res.read()
        data = src.decode('utf-8')
        html = BeautifulSoup(data, 'html.parser')
        tit_tag = html.select('') # 제목 추출
        con_tag = html.select('') # 내용 추출
        ans_tag = html.select('') # 답변 추출
    except:
        pass

titles = []
contents = []
answers = []

for i in tit_tag :
    tit = str(i.string)
    titles.append(tit.strip())
for i in con_tag :
    con = str(i.string)
    contents.append(con.strip())
for i in ans_tag :
    ans = str(i.string)
    contents.append(ans.strip())

