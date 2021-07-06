# 필요한 라이브러리 불러오기
import nltk
from nltk.data import normalize_resource_name
nltk.download('punkt')
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# 파일을 읽기모드로 열기
targetXML = open('ted_en-20160408.xml', 'r', encoding = 'UTF8')
# XML파일을 파싱하기
target_text = etree.parse(targetXML)

# 줄바꿈으로 구분을 해서 문자열로 바꾸어 반환
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# (Audio), (Laughter)등의 배경음(괄호) 부분을 제거
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 문장을열을 여러개의 조각(토큰)들로 쪼갬
sent_text = sent_tokenize(content_text)

normalized_text = []
for string in sent_text:

  # 정규 표현식으로 찾은 결과를 소문자로 바꾼 후 tokens에 저장
  tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())

  # normalized_text에 tokens를 추가
  normalized_text.append(tokens)

# 하나의 문장을 단어 단위로 나눈것을 result라는 리스트에 저장
result = [word_tokenize(sentence) for sentence in normalized_text]

# print('총 샘플의 개수 : {}'.format(len(result)))

# 상위 샘플 3개 출력
'''
for line in result[:3]:
  print(line)
'''

'''
size = 워드 벡터의 특징 값(임베딩 된 벡터의 차원)
window = 컨텍스트(문맥) 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습 X)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram
CBOW : 중앙 단어의 앞 뒤 단어를 고려해서 중앙에 있는 단어를 예츠하는 방법
Skip-gram : 중앙에 있는 단어를 통해서 중앙 단어의 앞 뒤 단어들을 예측하는 방법
'''
model = Word2Vec(sentences = result, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# man과 가장 유사한 단어
model_result = model.wv.most_similar("man")
# print(model_result)

from gensim.models import KeyedVectors
# 모델 저장
model.wv.save_word2vec_format('eng_w2v')
# 모델 로드
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")

# man과 가장 유사한 단어
model_result = loaded_model.most_similar('man')
# print(model_result)

from glove import Corpus, Glove

corpus = Corpus() 
corpus.fit(result, window=5)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

model_result1 = glove.most_similar("man")
print(model_result1)