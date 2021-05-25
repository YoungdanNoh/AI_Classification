# -*- coding: utf-8 -*- 
import os, json, glob, sys, numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Input, Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pymysql
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
import time

def Model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
    conn = engine.connect()

    data = pd.read_sql_table('after_prepro', conn) #db데이터를 csv파일처럼 읽어온다.
    
    print('a')
    data = data.drop_duplicates(['Content'], keep=False) # 중복되는 모든 값 제거
    print('b')
    
    df2 = data.sample(frac=1).reset_index(drop=True) #frac은 특정 비율로 데이터를 랜덤하게 샘플링 해옴
    print(df2.iloc[0:10,1])
    
    print(len(df2.iloc[:, 0]))
    
    X = df2.iloc[:, 0].values #뉴스 제목과 내용을 합친 데이터
    y = df2.iloc[:, 1].values #뉴스 카테고리 데이터
    print(y)
    
    print(len(X), len(y))
    
    nb_classes = len(set(y)) #카테고리의 갯수를 센다
    print(nb_classes)
    y = np_utils.to_categorical(y, nb_classes) #One-hot 인코딩하기
    print(y)
    
    
    max_word = 5000 #가장 많이 사용된 단어 5000개 사용
    max_len = 500 #길이는 500으로 제한
    
    tok = Tokenizer(num_words = max_word) #문자열을 여러개의 조각으로 나눈다.
    tok.fit_on_texts(X)
    print(len(tok.word_index))
    
    sequences = tok.texts_to_sequences(X) #텍스트를 시퀀스화 시켜준다.
    print(len(sequences[0]))
    print(sequences[0])
    print(len(tok.word_index))
    
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len) #sequences matrix로 만들어준다.
    print(sequences_matrix)
    print(sequences_matrix[0])
    print(len(sequences_matrix[0]))
    
    print(len(tok.word_index))
    
    X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y, test_size=0.2) 
    #8:2비율로 train과 test데이터를 나눠준다.
    
    print(X_train.shape)
    print(y_train.shape)
    print(len(X_train))
    print(len(X_test))
    
    #LSTM
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(LSTM(60, return_sequences=True)) #LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=7, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
      
    evaluate = model.evaluate(X_test, y_test)[1]
    correct = round(evaluate*len(X_test))
    #print(correct,'/',len(X_test))
    print("맞은갯수 : {}/{}".format(int(correct),len(X_test)))
      
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
      
#     sql = """delete from result"""
#     curs.execute(sql)
#       
#     sql = """insert into result (evaluate,correct,all_data,model) values (%s, %s,%s)"""
#     curs.execute(sql, (evaluate,int(correct),len(X_test)))
      
    # 모델을 사용해 예측해보고 test데이터와 비교하기 
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
              
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
      
    sql = """delete from text_analysis_result"""
    curs.execute(sql)
      
    sql = """insert into text_analysis_result (evaluate,Category,all_data,correct_data,model) values (%s, %s,%s,%s,%s)"""
      
      
    Category = ['생활문화','세계','IT과학']
    all_data = [all_lifeCulture, all_world, all_IT]
    correct_data = [lifeCulture, world, IT]
      
    for i in range(len(Category)): # 카테고리별 예측 결과 DB에 insert
        curs.execute(sql, (evaluate, Category[i], all_data[i], correct_data[i],'LSTM'))
          
    conn.commit()
    conn.close()
    
    #RNN
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(SimpleRNN(60, return_sequences=True)) #LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
      
    evaluate = model.evaluate(X_test, y_test)[1]
    correct = round(evaluate*len(X_test))
    #print(correct,'/',len(X_test))
    print("맞은갯수 : {}/{}".format(int(correct),len(X_test)))
      
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
      
#     sql = """delete from result"""
#     curs.execute(sql)
      
#     sql = """insert into result (evaluate,correct,all_data) values (%s, %s,%s)"""
#     curs.execute(sql, (evaluate,int(correct),len(X_test)))
      
    # 모델을 사용해 예측해보고 test데이터와 비교하기 
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
              
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
      
#     sql = """delete from text_analysis_result"""
#     curs.execute(sql)
      
    sql = """insert into text_analysis_result (evaluate,Category,all_data,correct_data,model) values (%s, %s,%s,%s,%s)"""
      
      
    Category = ['생활문화','세계','IT과학']
    all_data = [all_lifeCulture, all_world, all_IT]
    correct_data = [lifeCulture, world, IT]
      
    for i in range(len(Category)): # 카테고리별 예측 결과 DB에 insert
        curs.execute(sql, (evaluate, Category[i], all_data[i], correct_data[i], 'RNN'))
          
    conn.commit()
    conn.close()
    
    time.sleep(30)
    #Bi-LSTM
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(Bidirectional(LSTM(60, return_sequences=True))) #Bi-LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
      
    evaluate = model.evaluate(X_test, y_test)[1]
    correct = round(evaluate*len(X_test))
    #print(correct,'/',len(X_test))
    print("맞은갯수 : {}/{}".format(int(correct),len(X_test)))
      
    conn = pymysql.connect(host='127.0.0.1', user='DAN', password='dudeks7052', db='AI', charset='utf8')
    curs = conn.cursor()
    conn.commit()
      
#     sql = """delete from result"""
#     curs.execute(sql)
      
#     sql = """insert into result (evaluate,correct,all_data) values (%s, %s,%s)"""
#     curs.execute(sql, (evaluate,int(correct),len(X_test)))
      
    # 모델을 사용해 예측해보고 test데이터와 비교하기 
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
              
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
      
#     sql = """delete from text_analysis_result"""
#     curs.execute(sql)
      
    sql = """insert into text_analysis_result (evaluate,Category,all_data,correct_data,model) values (%s, %s,%s,%s,%s)"""
      
      
    Category = ['생활문화','세계','IT과학']
    all_data = [all_lifeCulture, all_world, all_IT]
    correct_data = [lifeCulture, world, IT]
      
    for i in range(len(Category)): # 카테고리별 예측 결과 DB에 insert
        curs.execute(sql, (evaluate, Category[i], all_data[i], correct_data[i], 'BiLSTM'))
          
    conn.commit()
    conn.close()

def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    engine = create_engine('mysql+pymysql://DAN:dudeks7052@localhost/AI',convert_unicode=True)
    conn = engine.connect()

    data = pd.read_sql_table('after_prepro', conn) #db데이터를 csv파일처럼 읽어온다.
    
    print('a')
    data = data.drop_duplicates(['Content'], keep=False) # 중복되는 모든 값 제거
    print('b')
    
    df2 = data.sample(frac=1).reset_index(drop=True) #frac은 특정 비율로 데이터를 랜덤하게 샘플링 해옴
    print(df2.iloc[0:10,1])
    
    print(len(df2.iloc[:, 0]))
    
    X = df2.iloc[:, 0].values #뉴스 제목과 내용을 합친 데이터
    y = df2.iloc[:, 1].values #뉴스 카테고리 데이터
    print(y)
    
    print(len(X), len(y))
    
    nb_classes = len(set(y)) #카테고리의 갯수를 센다
    print(nb_classes)
    y = np_utils.to_categorical(y, nb_classes) #One-hot 인코딩하기
    print(y)
    
    
    max_word = 5000 #가장 많이 사용된 단어 5000개 사용
    max_len = 500 #길이는 500으로 제한
    
    tok = Tokenizer(num_words = max_word) #문자열을 여러개의 조각으로 나눈다.
    tok.fit_on_texts(X)
    print(len(tok.word_index))
    
    sequences = tok.texts_to_sequences(X) #텍스트를 시퀀스화 시켜준다.
    print(len(sequences[0]))
    print(sequences[0])
    print(len(tok.word_index))
    
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len) #sequences matrix로 만들어준다.
    print(sequences_matrix)
    print(sequences_matrix[0])
    print(len(sequences_matrix[0]))
    
    print(len(tok.word_index))
    
    X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y, test_size=0.2) 
    #8:2비율로 train과 test데이터를 나눠준다.
    
    print(X_train.shape)
    print(y_train.shape)
    print(len(X_train))
    print(len(X_test))
    
    #LSTM
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(LSTM(60, return_sequences=True)) #LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
    
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
    
    print("LSTM")          
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
    
    time.sleep(30)
    
    #BiLSTM
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(Bidirectional(LSTM(60, return_sequences=True))) #LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
    
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
    
    print("BiLSTM")          
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
    
    time.sleep(30)
    
    #RNN
    with K.tf_ops.device('/device:GPU:0'):
        model = Sequential()
          
        model.add(Embedding(max_word, 64, input_length=max_len)) #단어를 임베딩 시킨다.
        model.add(SimpleRNN(60, return_sequences=True)) #LSTM모델을 쌓는다.
        model.add(GlobalMaxPool1D())
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu')) #모델 생성
        model.add(Dropout(1.0))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/Users/noyeongdan/Downloads/Server_AI/AISpring2/src/main/webapp/WEB-INF/views/model3'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/lstm.model"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
          
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
      
    model.summary()
      
    hist = model.fit(X_train, y_train, batch_size=250, epochs=5, validation_split=0.2, callbacks=[checkpoint, early_stopping])
      
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
    
    xhat_idx=[]
    print("xhat_idx: ",xhat_idx)
    for i in range(len(X_test)): # X_test의 모든 인덱스를 가져온다.
        xhat_idx.append(i)
          
    xhat = X_test[xhat_idx] # X_test의 모든 값들을 xhat에 넣어준다.
    yhat = model.predict_classes(xhat) # xhat의 값을 model을 사용해 예측하고 예측값을 yhat에 넣는다.
      
    correct = 0 # 예측이 성공한 갯수
      
    all_lifeCulture=0 # test데이터 중 생활문화 카테고리의 전체 갯수
    all_world=0 # test데이터 중 세계 카테고리의 전체 갯수
    all_IT=0 # test데이터 중 IT과학 카테고리의 전체 갯수
      
    lifeCulture=0 # test데이터 예측 후 예측이 성공한 생활문화 카테고리의 갯수
    world=0 # test데이터 예측 후 예측이 성공한 세계 카테고리의 갯수
    IT=0 # test데이터 예측 후 예측이 성공한 IT과학 카테고리의 갯수
    for i in range(len(X_test)):
        #print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
        # 실제 값과 예측값을 출력한다.
        if np.argmax(y_test[xhat_idx[i]]) == yhat[i]: # 예측 성공 시 correct값 1증가
            correct+=1
              
        if np.argmax(y_test[xhat_idx[i]])==0: # test데이터의 종류가 생활문화일 때 증가 
            all_lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==1: # test데이터의 종류가 세계일 때 증가 
            all_world+=1
        elif np.argmax(y_test[xhat_idx[i]])==2: # test데이터의 종류가 IT과학일 때 증가 
            all_IT+=1
            
        if np.argmax(y_test[xhat_idx[i]])==yhat[i]==0: # 생활문화 카테고리 예측에 성공 시 증가 
            lifeCulture+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==1: # 세계 카테고리 예측에 성공 시 증가 
            world+=1
        elif np.argmax(y_test[xhat_idx[i]])==yhat[i]==2: # IT과학 카테고리 예측에 성공 시 증가 
            IT+=1
    
    print("RNN")          
    print("전체 test데이터갯수: {0}".format(len(X_test)))
    print("correct : {0}".format(correct))
      
    print("생활문화: {0}/{1}".format(lifeCulture,all_lifeCulture))
    print("세계: {0}/{1}".format(world,all_world))
    print("IT과학: {0}/{1}".format(IT,all_IT))
    
    