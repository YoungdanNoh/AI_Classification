Spring과 Python을 사용한 텍스트 분류
===============================

* Mecab을 사용해 형태소분석을 하고 RNN, LSTM, BiLSTM을 사용해 훈련을 하며 모델을 생성한다.
* 훈련 결과를 Spring을 통해 시각적으로 제공한다.
* 실험 결과 주피터와 STS에서 실행하는 것에서 정확도, 훈련 시간, 손실값(val_loss, val_acc, train_loss, train_acc)에 약간의 차이가 있었다.
* 성능 비교는 STS에서 실행한 기준으로 비교하였다.
* 또한 매 훈련마다 정확도, 훈련 시간, 손실값(val_loss, val_acc, train_loss, train_acc)이 달라지지만 대부분 소수점 셋째자리에서 차이를 보일뿐 큰 차이는 보이지 않는다.

Version of this program
------------------------

* tf.__version__

  1.13.1
  
* import numpy

  numpy.version.version

  1.16.1
  
* keras.__version__

  2.3.1

* pip install tensorflow-gpu==2.0

<br/>

> ## RNN, LSTM, BiLSTM
>> 세 가지 모델의 텍스트 카테고리 분류 성능을 비교한다.

<img width="771" alt="스크린샷 2021-05-26 오전 1 03 57" src="https://user-images.githubusercontent.com/38938145/119530835-57d5d700-bdbe-11eb-9502-36edc3c225cf.png">

<img width="374" alt="스크린샷 2021-05-26 오전 1 04 13" src="https://user-images.githubusercontent.com/38938145/119530928-70de8800-bdbe-11eb-9a52-c27763274a94.png">
