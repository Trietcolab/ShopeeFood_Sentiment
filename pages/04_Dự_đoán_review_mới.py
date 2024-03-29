import streamlit as st
import pandas as pd
import numpy as np
import pickle
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex


st.set_page_config(page_title="Mô tả dự án")

st.markdown("# <center>Project 1:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Sentiment Analysis</span></center>", unsafe_allow_html=True)

st.subheader("Nhập vào comment", divider='rainbow')
# st.title('Nhập vào comment')
comment1 = st.text_input("Text 1")
comment2 = st.text_input("Text 2")
lst = [comment1, comment2]

# Chuyển comment vào dataframe
new_df = pd.DataFrame({"New_Comment": lst})
st.dataframe(new_df, width=500, height=100)

## Xử lý tiếng Việt
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
#print(teen_dict)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
#print(teen_dict)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
englist_lst = file.read().split('\n')
for line in englist_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
#print(teen_dict)
file.close()
################
#LOAD wrong words. Bỏ 1 số từ liên quan đến "ngon".
file = open('files/wrong-word2.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

# Tạo hàm xử lý emoji, teencode, translate, wrong words, stopwords
def process_text(text, dict_emoji, dict_teen, lst_wrong):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(dict_emoji[word]+' ' if word in dict_emoji else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(dict_teen[word] if word in dict_teen else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words
        sentence = ' '.join('' if word in lst_wrong else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

## Tiền xử lý Tiếng Việt cho comment mới
# Xử lý emoji, teencode, wrong word
new_df = new_df.assign(Comment_2=lambda x: x['New_Comment'].apply(lambda x: process_text(str(x), emoji_dict, teen_dict, wrong_lst)))
# Chuẩn hóa unicode
new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  covert_unicode(str(x))))
# Chuẩn hóa ký tự lặp
new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  normalize_repeated_characters(str(x))))
# Loại bỏ stopword
new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  remove_stopword(str(x), stopwords_lst)))

## Load dataframe sau khi xử lý
st.subheader("Comment sau khi xử lý tiếng Việt", divider='rainbow')
st.dataframe(new_df, width=500, height=100)

def predict_comment(df):
  ## Load model TfIDF đã save
  # Đường dẫn file model Tfidf
  path_tfidf = "data/tfidf02_model.pkl"
  # Load model
  with open(path_tfidf, 'rb') as f:
    tfidf_model = pickle.load(f)
  ## Transform data comment mới
  # Tạo dataframe mới chứa các tfidf features
  df_tfidf02 = pd.DataFrame(tfidf_model.transform(df['Comment_2']).toarray(), columns=tfidf_model.get_feature_names_out())
  st.write("##### :blue[Dataframe đã xử lý tfidf]")
  st.dataframe(df_tfidf02)
  
  ## Load model Logistic Regression
  # Đường dẫn file model Logistic Regression  
  path_lr = "data/lr_smote_model.pkl"
  # Load model
  with open(path_lr, 'rb') as f:
    lr_model = pickle.load(f)
    
  ## Dự đoán comment mới
  # Dự đoán
  pred = lr_model.predict(df_tfidf02)
  # Chuyển kết quả vào dataframe
  new_df["pred"] = pred

  new_df["Đánh giá"] = new_df["pred"].map({0: "Tiêu cực", 1: "Tích cực"})
  ## Load dataframe sau khi xử lý
  st.subheader("Kết quả dự đoán", divider='rainbow')
  st.dataframe(new_df[["New_Comment", "Đánh giá"]],
               column_config={
                "New_Comment": "Bạn đã nhập",
                },
               hide_index=True, width=500, height=100)
  
if new_df.size != 0:
  predict_comment(new_df)

# st.title('Kết quả dự đoán')
# # defining random values in a dataframe using pandas and numpy
# df = pd.DataFrame({
#     "Bạn đã nhập": ["Thịt nướng ngon lắm!", "Nhân viên phục vụ cáu với khách"],
#     "Đánh giá": ["Tích cực", "Tiêu cực"]
# })

# https://stackoverflow.com/questions/68379442/how-to-use-python-dataframe-styling-in-streamlit
# style
# th_props = [
#   ('font-size', '14px'),
#   ('text-align', 'center'),
#   ('font-weight', 'bold'),
#   ('color', '#6d6d6d'),
#   ('background-color', '#f7ffff')
#   ]
                               
# td_props = [
#   ('font-size', '12px')
#   ]
                                 
# styles = [
#   dict(selector="th", props=th_props),
#   dict(selector="td", props=td_props)
#   ]

# # table
# df2=df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
# st.dataframe(df2, hide_index=True)