import csv
import requests
from bs4 import BeautifulSoup
import time

# 웹 페이지 URL 설정
base_url = 'http://m.ilbe.com/search?docType=doc&searchType=&page=1&q=%EC%A2%86%EA%B0%99%EB%84%A4'

# User-Agent 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# CSV 파일을 열고 데이터를 쓰기 모드로 연다
with open('scraped_data_jotgantne.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['Number', 'Content'])

    # 웹 스크래핑 및 데이터를 파일에 쓰기
    for page in range(1, 20):  # 페이지 범위 설정
        url = base_url.format(page)
        success = False
        retries = 0
        while not success and retries < 5:  # 최대 5번 재시도
            try:
                response = requests.get(url, headers=headers, verify=False)  # SSL 인증서 검증 비활성화
                soup = BeautifulSoup(response.content, 'html.parser')

                # item-box__article 클래스의 모든 요소 선택
                content_list = soup.find_all(class_='item-box__article')

                for i, content in enumerate(content_list, start=1):
                    text = content.get_text(strip=True)
                    writer.writerow([(page - 1) * 20 + i, text])  # 페이지 번호와 텍스트 저장

                success = True  # 성공하면 루프를 벗어남

            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                time.sleep(10)  # 실패 시 10초 대기

        # 요청 간 대기 시간 추가
        time.sleep(10)  # 10초 대기

print("스크래핑 완료.")
