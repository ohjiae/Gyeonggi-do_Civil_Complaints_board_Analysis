import urllib.request as req # 원격 서버 url 자료 요청 
from bs4 import BeautifulSoup # source -> html 파싱 
import re


'''
url 쿼리를 이용한 새로운 titles 생성 코드입니다.
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

# 4. 고유 주소 넘버를 for문으로 돌면서 제목, 답변을 긁어오는 코드(미완성)
titles = []
contents = []
answers = []

'''
아이디어] address1, 2.. 이런식으로 쪼개서 한 1000개씩 긁어오는거 확인할 수 있도록 만들까 고민중 에러 나오는데 있으면 그 구간 버리는 형식으로
'''

for i in address:
    try:
        res = req.urlopen(base_url2+"?_csrf=eea0ef35-d28a-4e60-ad98-e92df36b6643&recordCountPerPage=1&pageIndex=1&epUnionSn=%s&dutySctnNm=taol&lcgovBlngInstCd=&searchWordType=1&searchWord=&rqstStDt=2020-07-31&rqstEndDt=2021-07-31&dateType=0&pttnTypeNm=&searchInstType=locgovDiv&searchInstCd=6410000&focusPerPageYn=&frm_frmMenuMngNo=&frm_instCd=&frm_frmUrlSn=&frm_frmAllUrl="% i)
        src = res.read()
        data = src.decode('utf-8')
        html = BeautifulSoup(data, 'html.parser')
        tit = html.select('#txt > div.same_mwWrap > div.samBox.mw > div > div > strong') # 제목 추출
        ans = html.select('#txt > div.same_mwWrap > div.samBox.ans > div > div.samC_top') # 답변 추출
        titles.append(str(tit[0].string))
        answers.append(str(ans))     
    except:
        pass



