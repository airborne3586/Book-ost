import streamlit as st

st.markdown('## 1️⃣ 필요한 모듈 설치')
code ='''
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice

import requests
import json

from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By

import time

from googletrans import Translator

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
'''

st.code(code, language='python')
st.divider()
st.markdown('## 2️⃣ 도서 정보 들고오기')
code = '''
data = pd.read_excel('final_data.xlsx',index_col=0)

rest_api_key = "41d651c93152d5ec054dc828cacfa671"
url = "https://dapi.kakao.com/v3/search/book"
header = {"authorization": "KakaoAK "+rest_api_key}

#궁금한 도서의 isbn 입력
querynum = {"query": '어린왕자'}

#도서 정보 불러오기
response = requests.get(url, headers=header, params = querynum)
content = response.text
책정보 = json.loads(content)['documents'][0]

book = pd.DataFrame({'title': 책정보['title'],
              'isbn': 책정보['isbn'],
              'authors': 책정보['authors'],
              'publisher': 책정보['publisher']})
book
'''
st.code(code, language='python')
st.image('어린왕자.png')

st.markdown('  ')
st.markdown('### 도서 정보 크롤링')
code = '''
target_url = 책정보['url']
# 옵션 생성
options = webdriver.ChromeOptions()
# 창 숨기는 옵션 추가
options.add_argument("headless")

driver = webdriver.Chrome(options=options)
driver.get(target_url)
time.sleep(5)

try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/div[3]/a')
    botton.click()
except :
    pass
책소개 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/p')

time.sleep(3)
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/div[3]/a')
    botton.click()
except :
    pass
책속으로 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/p')

time.sleep(3)
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/div[3]/a')
    botton.click()
except :
    pass
서평 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/p')

book['책소개'] = 책소개.text
book['책속으로'] = 책속으로.text
book['서평'] = 서평.text

driver.close()
'''
st.code(code, language='python')

st.markdown('### 텍스트 전처리')
code = '''
#영어 불용어 사전
stops = set(stopwords.words('english'))

def hapus_url(text):
    mention_pattern = r'@[\w]+'
    cleaned_text = re.sub(mention_pattern, '', text)
    return re.sub(r'http\S+','', cleaned_text)

#특수문자 제거
#영어 대소문자, 숫자, 공백문자(스페이스, 탭, 줄바꿈 등) 아닌 문자들 제거
def remove_special_characters(text, remove_digits=True):
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


#불용어 제거
def delete_stops(text):
    text = text.lower().split()
    text = ' '.join([word for word in text if word not in stops])
    return text
   
    
#품사 tag 매칭용 함수
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

#품사 태깅 + 표제어 추출
def tockenize(text):
    tokens=word_tokenize(text)
    pos_tokens=nltk.pos_tag(tokens)
    
    text_t=list()
    for _ in pos_tokens:
        text_t.append([_[0], get_wordnet_pos(_[1])])
    
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word[0], word[1]) for word in text_t])
    return text



def clean(text):
    text = remove_special_characters(text, remove_digits=True)
    text = delete_stops(text)
    text = tockenize(text)
    return text
'''
st.code(code, language='python')

code = '''
translator = Translator()
for col in ['책소개', '책속으로', '서평']:
    name = col+'_trans'
    if book[col].values == '':
        book[name] = ''
        continue
    book[name] = clean(translator.translate(hapus_url(book.loc[0, col])).text)

    book
'''
st.code(code, language='python')
st.image('트랜스.png')

code = '''total_text = book.loc[0, '책소개_trans'] + book.loc[0, '책속으로_trans'] + book.loc[0, '서평_trans']
total_text'''
st.code(code, language='python')
st.markdown('beautiful story little prince love world adult contradict...')

st.markdown('### 도서에서 감정 추출')
st.caption('트위터 데이터로 학습시킨 SVM모델을 가지고 도서정보에서 감정을 추출합니다.')
code = '''
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv('tweet_data_agumentation.csv', index_col = 0)

tfidf_vect_emo = TfidfVectorizer()
tfidf_vect_emo.fit_transform(df["content"])

model = joblib.load('SVM.pkl')
total_text2 = tfidf_vect_emo.transform(pd.Series(total_text))
model.predict_proba(total_text2)
sentiment = pd.DataFrame(model.predict_proba(total_text2),index=['prob']).T
sentiment['감정'] = ['empty','sadness','enthusiasm','worry','love','fun','hate','happiness','boredom','relief','anger']
'''
st.code(code, language='python')
st.image('감정.png')
st.divider()
st.markdown('## 3️⃣ audio feature와 text 유사도')
st.caption('감정적 특성 유사도 : 트위터 감정 데이터를 이용해 학습한 SVM 모델로 도서 설명 데이터와 노래 가사에서 감정을 추출하고, 노래 audio feature에서 추출한 감정과 유사도를 계산합니다.')
code = '''
audio_data = data.iloc[:,-12:-1]

sentiment_prob = sentiment['prob']
sentiment_prob.index = sentiment['감정']

audio_data.columns = ['empty', 'sadness', 'enthusiasm', 'worry', 'love', 'fun', 'hate',
       'happiness', 'boredom', 'relief', 'anger']

audio_data_1 = pd.concat([sentiment_prob,audio_data.T],axis=1).T

col = ['book']+list(data['name'])
cosine_sim_audio = cosine_similarity(audio_data_1)
cosine_sim_audio_df = pd.DataFrame(cosine_sim_audio, index = col, columns=col)

audio_sim = cosine_sim_audio_df['book']
'''
st.code(code, language='python')
st.divider()
st.markdown('## 4️⃣ 가사와 text 유사도')
code = '''
lyrics_data = data.iloc[:,5:-12]

lyrics_data_1 = pd.concat([sentiment_prob,lyrics_data.T],axis=1).T

cosine_sim_lyrics = cosine_similarity(lyrics_data_1)
cosine_sim_lyrics_df = pd.DataFrame(cosine_sim_lyrics, index =col, columns=col)

lyrics_sim = cosine_sim_lyrics_df['book']
'''
st.code(code, language='python')
st.divider()
st.markdown('## 5️⃣ 키워드와 text 유사도')
st.caption('내용 유사도 : 도서 설명 데이터와 노래 가사에 TF-IDF 방식을 사용하여 유사도를 계산합니다.')
code = '''
keyword_data = data['key_word']
book_song_cont1 = pd.DataFrame({"text": total_text}, index = range(1))
book_song_cont2 = pd.DataFrame({"text": keyword_data})
keyword_data_1 = pd.concat([book_song_cont1, book_song_cont2], axis=0).reset_index(drop=True)

tfidf_vect_cont = TfidfVectorizer()
tfidf_matrix_cont = tfidf_vect_cont.fit_transform(keyword_data_1['text'])
tfidf_array_cont = tfidf_matrix_cont.toarray()
tfidf_df_cont = pd.DataFrame(tfidf_array_cont, columns=tfidf_vect_cont.get_feature_names_out())

cosine_sim_keyword = cosine_similarity(tfidf_array_cont)
cosine_sim_keyword_df = pd.DataFrame(cosine_sim_keyword, index = col, columns=col)

keyword_sim = cosine_sim_keyword_df['book']
'''
st.code(code, language='python')

st.divider()
st.markdown('## 6️⃣ 전체 유사도')
st.markdown('### AF : 가사 : 키워드 = 0.8 : 0.1 : 0.1')
code = '''
total_sim  = 0.8*audio_sim + 0.1*lyrics_sim + 0.1*keyword_sim

recommend_song = total_sim.sort_values(ascending=False)[1:6].index
total_sim_df = pd.DataFrame(total_sim[1:])
total_sim_df = total_sim_df.reset_index()
total_sim_df.columns = ['name','book']

top_five = total_sim_df.sort_values(by='book',ascending=False)[:5]
index = total_sim_df.sort_values(by='book',ascending=False)[:5].index.sort_values()

artist = data.iloc[index][['url','name','Artist']]
top_five_df = pd.merge(artist,top_five,on='name').sort_values(by='book',ascending=False).drop_duplicates()
'''
st.code(code, language='python')

st.markdown('### AF : 가사 : 키워드 = 0 : 0.5 : 0.5')
code = '''
total_sim  = 0*audio_sim + 0.5*lyrics_sim + 0.5*keyword_sim

recommend_song = total_sim.sort_values(ascending=False)[1:6].index
total_sim_df_1 = pd.DataFrame(total_sim[1:])
total_sim_df_1 = total_sim_df_1.reset_index()
total_sim_df_1.columns = ['name','book']

top_five_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5]
index_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5].index.sort_values()

artist = data.iloc[index_1][['url','name','Artist']]
top_five_df_1 = pd.merge(artist,top_five_1,on='name').sort_values(by='book',ascending=False).drop_duplicates()
'''
st.code(code, language='python')
