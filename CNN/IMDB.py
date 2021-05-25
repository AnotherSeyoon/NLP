from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words = vocab_size)
'''
print(X_train[:5])
'''
# 패딩
max_len = 200
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

# 패딩되었는지 확인
'''
print('X_train의 크기(shape) :', X_train.shape)
print('X_test의 크기(shape) :', X_test.shape)

print(y_train)
'''

from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# 모델 설계
model = Sequential() # 순차 모델로 만듬
model.add(Embedding(vocab_size, 256)) # 범주형 자료를 연속형 벡터로 치환
model.add(Dropout(0.3)) # 신경망 모델이 복잡해질 때 뉴런의 연결을 임의로 삭제
model.add(Conv1D(256, 3, padding = 'valid', activation = 'relu')) # Conv1D 레이어 배치, 커널수는 256, 커널의 크기는 3
model.add(GlobalMaxPooling1D()) # 여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환
model.add(Dense(128, activation = 'relu')) # 입력과 출력을 모두 연결해줌, 출력 누런의 수, relu(rectifier) 함수 : 은익층에 주로 쓰임
model.add(Dropout(0.5)) 
model.add(Dense(1, activation = 'sigmoid')) # 출력 뉴런의 수, sigmoid(시그모이드) 함수 : 이진 분류 문제에서 출력층에 주로 쓰임

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3) # 검증 데이터 손실이 3회 증가하면 학습을 중단하는 조기종료(Earlystopping)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True) # 검증 데이터의 정확도가 이전보다 좋아질 경우, 모델 저장

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test), callbacks = [es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))