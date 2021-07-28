# ITWILL-Final_project <민원 게시판>


## 변수 목록
### 기본 세팅용 변수
- 현재 파일 경로
# pwd = = os.getcwd()
- 크롬 드라이버 가져오기
driver = webdriver.Chrome("")
- 크롬 드라이버 10초 기다리기 (로딩 대기용)
wait = WebDriverWait(driver, 10) 
- xpath로 요소 찾기
find_by_xpath = driver.find_elements_by_xpath

### 메인 변수
post_list = 게시글을 모아둔 리스트
