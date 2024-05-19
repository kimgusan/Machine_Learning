# from faker import Faker
# import pandas as pd
# import random
#
# # 한글 데이터 생성
# fake = Faker('ko_KR')
#
#
# # 더미 데이터 생성 함수
# def create_dummy_data(num_records):
#     data = []
#     for _ in range(num_records):
#         title = fake.sentence(nb_words=6)  # 자연스러운 한글 제목
#         description = ' '.join([fake.sentence() for _ in range(3)])  # 자연스러운 한글 설명
#
#         record = {
#             'id': random.randint(1, 1000),
#             'created_at': fake.date_time_this_year(),
#             'updated_at': fake.date_time_this_year(),
#             'title': title,
#             'description': description,
#             'image1': f"club/{fake.date_this_year().strftime('%Y/%m/%d')}/profile_{random.randint(1, 10)}.jpg",
#             'image2': f"club/{fake.date_this_year().strftime('%Y/%m/%d')}/bicycle_{random.randint(1, 10)}.jpg",
#             'status': random.randint(1, 3),
#             'likes': random.randint(0, 100)
#         }
#         data.append(record)
#     return pd.DataFrame(data)
#
#
# # 1000개의 더미 데이터 생성
# dummy_data = create_dummy_data(1000)
#
# # 더미 데이터 출력
# print(dummy_data.head())
#
# # CSV 파일로 저장
# dummy_data.to_csv('t_random.csv', index=False, encoding='utf-8-sig')
from datetime import datetime

from faker import Faker
import pandas as pd
import random

# 한글 데이터 생성
fake = Faker('ko_KR')


# 더미 데이터 생성 함수
def create_dummy_data(num_records):
    data = []
    for _ in range(num_records):
        title = fake.sentence(nb_words=6)  # 자연스러운 한글 제목
        description = ' '.join([fake.sentence() for _ in range(3)])  # 자연스러운 한글 설명

        record = {
            # 'id': random.randint(1, 1000),
            'created_at': fake.date_time_this_year(),
            'updated_at': datetime.now(),  # 현재 시간으로 설정
            'video_path': f"teenplay_video/{fake.date_this_year().strftime('%Y/%m/%d')}/travel_video.MOV",
            'thumnail_path': f"teenplay_thumbnail/{fake.date_this_year().strftime('%Y/%m/%d')}/travel_thumnail.png"
        }
        data.append(record)
    return pd.DataFrame(data)


# 1000개의 더미 데이터 생성
dummy_data = create_dummy_data(500)

# 더미 데이터 출력
print(dummy_data.head())

# CSV 파일로 저장
dummy_data.to_csv('t_travel.csv', index=False, encoding='utf-8-sig')

# another,o
# culture,o
# town,
# food,o
# hobby,o
# love 0,
# sport,o
# growth 0,
# travel,o

