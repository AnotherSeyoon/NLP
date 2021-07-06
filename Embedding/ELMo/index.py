# 필요한 라이브러리 임포트
import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

# 텐서플로우 허브로 부터 ELMo를 다운로드
elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# 스팸 메일 분류하기 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding = 'latin-1')
data[:5]