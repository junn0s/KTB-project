import streamlit as st
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

conn = pymysql.connect(
        host='',
        port=9999,
        user='avnadmin',
        password='',
        database='cnn_training_log',
        ssl={'ca': '/content/ca.pem'},
        charset='utf8mb4'
)
# 학습 로그 불러오기
df = pd.read_sql("SELECT * FROM cnn_training_log ORDER BY epoch", conn)
conn.close()

st.title("ShuffleNet_0.5 학습 대시보드")

col1, col2 = st.columns([1, 1])

with col1:
    st.line_chart(df[['train_accuracy', 'val_accuracy']])

with col2:
    st.line_chart(df[['train_loss', 'val_loss']])

st.dataframe(df, use_container_width=True)
