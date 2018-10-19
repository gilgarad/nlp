# Author: Isaac Sim
# Modified Version from: https://beomi.github.io/2017/01/20/HowToMakeWebCrawler/

# class WebCrawler:


# import requests
# from bs4 import BeautifulSoup
# import json
# import os
#
# # python파일의 위치
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# base_url = 'https://cafe.naver.com/sevenknights?iframe_url=/ArticleList.nhn%3Fsearch.clubid=26903084%26search.menuid=104%26search.boardtype=L'
# req = requests.get(base_url)
# html = req.text
# soup = BeautifulSoup(html, 'html.parser')
# my_titles = soup.select(
#     'h3 > a'
#     )
#
# data = {}
#
# for title in my_titles:
#     data[title.text] = title.get('href')
#     print(title.text)
#     print(title.get('href'))
#
# with open(os.path.join(BASE_DIR, 'result.json'), 'w+') as json_file:
#     json.dump(data, json_file)


# from selenium import webdriver
# from bs4 import BeautifulSoup as bs
# import pandas as pd
# # chromedriver는 다운로드 후 경로 지정을 해줘야 한다. (현재는 같은 폴더 )
# driver = webdriver.Chrome('./chromedriver')
# driver.implicitly_wait(3)
#
# # 로그인 전용 화면
# driver.get('https://nid.naver.com/nidlogin.login')
# # 아이디와 비밀번호 입력
# driver.find_element_by_name('id').send_keys('gilgarada')
# driver.find_element_by_name('pw').send_keys('tladltkr1')
# # 로그인 버튼 클릭
# driver.find_element_by_css_selector('#frmNIDLogin > fieldset > input').click()
#
# # base_url = 'https://cafe.naver.com/카페명/ArticleList.nhn?search.clubid=***'
# base_url = 'https://cafe.naver.com/sevenknights?iframe_url=/ArticleList.nhn%3Fsearch.clubid=26903084%26search.menuid=104%26search.boardtype=L'
# driver.get(base_url + '&search.menuid=***&search.page=***')
# # iframe으로 프레임 전환
# driver.switch_to_frame('cafe_main')
#
# # href 속성을 찾아 url을 리스트로 저장한다.
# article_list = driver.find_elements_by_css_selector('span.aaa > a.m-tcol-c')
# article_urls = [ i.get_attribute('href') for i in article_list ]
#
# res_list = []
# # Beautifulsoup 활용
# for article in article_urls:
#     driver.get(article)
#     # article도 switch_to_frame이 필수
#     driver.switch_to_frame('cafe_main')
#     soup = bs(driver.page_source, 'html.parser')
#     # 게시글에서 제목 추출
#     title = soup.select('div.tit-box span.b')[0].get_text()
#     # 내용을 하나의 텍스트로 만든다. (띄어쓰기 단위)
#     content_tags = soup.select('#tbody')[0].select('p')
#     content = ' '.join([ tags.get_text() for tags in content_tags ])
#     # dict형태로 만들어 결과 list에 저장
#     res_list.append({'title' : title, 'content' : content})
#     # time.sleep 작업도 필요하다.
# # 결과 데이터프레임화
# cafe_df = pd.DataFrame(res_list)
# # csv파일로 추출
# cafe_df.to_csv('cafe_crawling.csv', index=False)

