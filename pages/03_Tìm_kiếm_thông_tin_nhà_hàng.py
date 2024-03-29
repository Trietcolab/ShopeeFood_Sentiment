import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Mô tả dự án",
                   layout="wide",
                   initial_sidebar_state="auto")

st.markdown("# <center>Project 1:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Sentiment Analysis</span></center>", unsafe_allow_html=True)


id = st.number_input("#### Nhập ID nhà hàng", value=None, min_value=1, placeholder="Gõ vào số từ 1...1612")

# Load dataframe comment đã xử lý tiếng Việt"
review_data_10 = pd.read_csv('data/2_Reviews_clean09.csv', encoding='utf8', sep=',')
# Load dataframe thông tin nhà hàng
res_data = pd.read_csv('data/1_Restaurants.csv', encoding='utf8', sep=',')
# Load dataframe percent rating
rating_percent = pd.read_csv('data/rating_percent.csv', encoding='utf8', sep=',')

def name_address(id):
    name = "".join(res_data[res_data["ID"]==id].loc[:, "Restaurant"].values.tolist())
    address = "".join(res_data[res_data["ID"]==id].loc[:, "Address"].values.tolist())
    price = "".join(res_data[res_data["ID"]==id].loc[:, "Price"].values.tolist())
    return name, address, price
name, address, price = name_address(id)

def rating_show(id):
    # Lọc rating theo ID nhà hàng
    rating_res = rating_percent[rating_percent["IDRestaurant"]==id]
    if rating_res.size == 0:
        st.write("## Không có Rating nào!")
    else:
        # Tính rating trung bình
        value_series = rating_res["count_rating"]
        weighted_sum = sum(value_series * range(11))
        sum_rate = rating_res["count_rating"].sum()
        avg_rating = round(weighted_sum / sum_rate, 2)
        st.write('### Average Rating: ', avg_rating, "/10")
        # Các level rating
        st.write('### Rating bar')
        st.dataframe(
            rating_res[["rating_group", "count_rating", "Rating_Percent"]],
            column_config={
                "rating_group": "Rating score",
                "count_rating": "Số lượng",
                "Rating_Percent": st.column_config.ProgressColumn("Percent %",
                                                                help="The Rating volume",
                                                                format="%f",
                                                                min_value=0,
                                                                max_value=100,
                                                                ),
                },
            hide_index=True,
        )
    
def comment_count(id):
    # Đếm số lượng comment mỗi loại
    cmt_count = review_data_10[review_data_10["IDRestaurant"]==id].groupby("Sentiment").count().loc[:, "Comment_2"].reset_index()
    if cmt_count.size == 0:
        st.write("## Không có Comment nào!")
        
    elif cmt_count.size == 4:
        cmt1 = cmt_count.iloc[0,0]
        cmt1_count = cmt_count.iloc[0,1]
        cmt2 = cmt_count.iloc[1,0]
        cmt2_count = cmt_count.iloc[1,1]
        st.write("### Số lượng comment")
        st.write(cmt1, ": ", cmt1_count, "----- Tỷ lệ: ", round(cmt1_count/(cmt1_count+cmt2_count)*100,1), "%")
        st.write(cmt2, ": ", cmt2_count, "----- Tỷ lệ: ", round(cmt2_count/(cmt1_count+cmt2_count)*100,1), "%")

        # Thời gian comment mỗi loại
        cmt1_filter = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==cmt1)]
        min_cmt1 = cmt1_filter["Time_2"].min()
        max_cmt1 = cmt1_filter["Time_2"].max()
        cmt2_filter = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==cmt2)]
        min_cmt2 = cmt2_filter["Time_2"].min()
        max_cmt2 = cmt2_filter["Time_2"].max()
        st.write("### Thời gian nhận comment")
        st.write(cmt1, ": ", " từ ngày ", min_cmt1, " đến ngày ", max_cmt1)
        st.write(cmt2, ": ", " từ ngày ", min_cmt2, " đến ngày ", max_cmt2)
    else:
        cmt1 = cmt_count.iloc[0,0]
        cmt1_count = cmt_count.iloc[0,1]
        
        st.write("### Số lượng comment")
        st.write(cmt1, ": ", cmt1_count, "----- Tỷ lệ: ", round(cmt1_count/(cmt1_count)*100,1), "%")
    
        # Thời gian comment mỗi loại
        cmt1_filter = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==cmt1)]
        min_cmt1 = cmt1_filter["Time_2"].min()
        max_cmt1 = cmt1_filter["Time_2"].max()
        st.write("### Thời gian nhận comment")
        st.write(cmt1, ": ", " từ ngày ", min_cmt1, " đến ngày ", max_cmt1)

def cmt_wordcloud(id):
    # Đếm số lượng comment mỗi loại
    cmt_count = review_data_10[review_data_10["IDRestaurant"]==id].groupby("Sentiment").count().loc[:, "Comment_2"].reset_index()
    if cmt_count.size == 0:
        no_cmt = st.write("## Không có WordCloud Comment nào!")
        return no_cmt
    
    elif cmt_count.size == 4:
        # Lọc comment tích cực
        pos = "Tích cực"
        cmt_pos = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==pos)].loc[:, "Comment_2"].reset_index()
        text_pos = ' '.join(cmt_pos['Comment_2'])
        # Lọc comment tiêu cực
        neg = "Tiêu cực"
        cmt_neg = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==neg)].loc[:, "Comment_2"].reset_index()
        text_neg = ' '.join(cmt_neg['Comment_2'])
        
        col1, _, col2 = st.columns((60,1,60))
        with col1:
            # Vẽ wordcloud comment tích cực
            wordcloud_pos = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(text_pos)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.axis('off')
            plt.title("Comment: Tích cực", fontsize = 40)
            plt.show()
            st.pyplot()
        with col2:
            # Vẽ wordcloud comment tiêu cực
            wordcloud_neg = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(text_neg)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.axis('off')
            plt.title("Comment: Tiêu cực", fontsize = 40)
            plt.show()
            st.pyplot()
    
    else:
        loai_cmt = cmt_count.iloc[0,0]
        cmt_ = review_data_10[(review_data_10["IDRestaurant"]==id) & (review_data_10["Sentiment"]==loai_cmt)].loc[:, "Comment_2"].reset_index()
        text_ = ' '.join(cmt_['Comment_2'])
        
        col1, col2, col3 = st.columns((5,50,5))
        with col1:
            col1.write(' '*5)
        with col2:
            # Vẽ wordcloud comment
            wordcloud_ = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(text_)
            plt.figure(figsize=(5, 2))
            plt.imshow(wordcloud_, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Comment: {loai_cmt}", fontsize = 30)
            plt.show()
            st.pyplot()
        with col3:
            col3.write(' '*5)
        
st.subheader("Thông tin nhà hàng", divider='rainbow')
st.write("Tên nhà hàng:  ", name)
st.write("Địa chỉ:  ", address)
st.write("Giá món:  ", price)

st.subheader("Rating & Comment", divider='rainbow')
col1, col2 = st.columns((60,60))
with col1:
    comment_count(id)
with col2:
    rating_show(id)

st.subheader("Comment Wordcloud", divider='rainbow')
st.set_option('deprecation.showPyplotGlobalUse', False)
cmt_wordcloud(id)
