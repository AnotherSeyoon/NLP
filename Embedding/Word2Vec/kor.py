from http.client import OK
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')
train_data[:5]

# 리뷰 개수 출력
# print('리뷰 개수: ', len(train_data))

# Null 값 존재 유무
# print('존재 여부: ', train_data.isnull().values.any())

# Null 값이 존재하는 행 제거
train_data = train_data.dropna(how = 'any')

# Null 값이 존재하는지 확인
# print('존재 여부: ',train_data.isnull().values.any())

# 리뷰 개수 출력
# print('리뷰 개수: ', len(train_data))

# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 상위 5개 출력
# print(train_data[:5])

# 불용어 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업
okt = Okt()
tokenized_data = []
for sentence in train_data['document']:
    
    # 토큰화
    temp_X = okt.morphs(sentence, stem = True)
    
    # 불용어 제거
    temp_X = [word for word in temp_X if not word in stopwords]

    tokenized_data.append(temp_X)

# 리뷰 길이 분포 확인
print('리뷰의 최대 길이: ', max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이: ', sum(map(len, tokenized_data)) / len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins = 50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

'''
size = 워드 벡터의 특징 값(임베딩 된 벡터의 차원)
window = 컨텍스트(문맥) 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습 X)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram
CBOW : 중앙 단어의 앞 뒤 단어를 고려해서 중앙에 있는 단어를 예츠하는 방법
Skip-gram : 중앙에 있는 단어를 통해서 중앙 단어의 앞 뒤 단어들을 예측하는 방법
'''
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape)

# "최민식"과 가장 유사한 단어
print(model.wv.most_similar("최민식"))

# "히어로"와 가장 유사한 단어
print(model.wv.most_similar("히어로"))