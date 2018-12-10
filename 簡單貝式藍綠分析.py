#!/usr/bin/env python
# coding: utf-8

# # 匯入資料 製作train,test

# In[127]:


import pandas as pd
import json
with open('blue_label/opinion_bw_news_10', 'r') as f:
    data1 = json.load(f)
datapd1 = pd.DataFrame(data1)

with open('blue_label/opinion_bw_news_20', 'r') as f:
    data2 = json.load(f)
datapd2 = pd.DataFrame(data2)

with open('blue_label/opinion_bw_news_30', 'r') as f:
    data3 = json.load(f)
datapd3 = pd.DataFrame(data3)

with open('blue_label/opinion_bw_news_40', 'r') as f:
    data4 = json.load(f)
datapd4 = pd.DataFrame(data4)

with open('blue_label/opinion_op_news_10', 'r') as f:
    data5 = json.load(f)
datapd5 = pd.DataFrame(data5)

with open('blue_label/opinion_op_news_20', 'r') as f:
    data6 = json.load(f)
datapd6 = pd.DataFrame(data6)

with open('blue_label/opinion_op_news_30', 'r') as f:
    data7 = json.load(f)
datapd7 = pd.DataFrame(data7)

with open('blue_label/opinion_op_news_40', 'r') as f:
    data8 = json.load(f)
datapd8 = pd.DataFrame(data8)

with open('green_label/opinion_liberty_coldeye_news_final', 'r') as f:
    data9 = json.load(f)
datapd9 = pd.DataFrame(data9)

with open('green_label/opinion_liberty_kc_news_10', 'r') as f:
    data10 = json.load(f)
datapd10 = pd.DataFrame(data10)

with open('green_label/opinion_liberty_kc_news_20', 'r') as f:
    data11 = json.load(f)
datapd11 = pd.DataFrame(data11)

with open('green_label/opinion_liberty_op_news_10', 'r') as f:
    data12 = json.load(f)
datapd12 = pd.DataFrame(data12)

with open('green_label/opinion_liberty_op_news_20', 'r') as f:
    data13 = json.load(f)
datapd13 = pd.DataFrame(data13)

with open('green_label/opinion_liberty_op_news_30', 'r') as f:
    data14 = json.load(f)
datapd14 = pd.DataFrame(data14)

with open('green_label/opinion_liberty_op_news_40', 'r') as f:
    data15 = json.load(f)
datapd15 = pd.DataFrame(data15)

train_df = pd.concat( [datapd1, datapd2, datapd3, datapd4, datapd5, datapd6, datapd7, datapd8, datapd9, datapd10, datapd11, datapd12, datapd13, datapd14, datapd15], axis=0 )

with open('talk_ltn_printed_all.js') as f:
    data16 = json.load(f)
test_df = pd.DataFrame(data16)


# # def結巴且將train,test跑過 

# In[128]:


import jieba
def cutflow(p):
    cutresult = " ".join(jieba.cut(p))
    return cutresult.replace("\r", "").replace("\n", "")
train_df["content"] = train_df["content"].apply(cutflow)
test_df["content"] = test_df["content"].apply(cutflow)


# # 去除不必要欄位

# In[129]:


train_df = train_df.drop(["date_","infor","title","type"],axis=1)
test_df = test_df.drop(["author","date_","head","type","title"],axis=1)


# # 詞頻率量化

# In[130]:


from sklearn.feature_extraction.text import CountVectorizer #詞頻率量化
vec = CountVectorizer()
counts = vec.fit_transform(train_df["content"])#fit:算有多少欄位，有幾種。     #trnasform:每個、每種詞，出現幾次。
test_counts = vec.transform(test_df["content"])


# # 跑 sklearn

# In[131]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(counts, train_df["tag"])
pre = clf.predict(test_counts)
from sklearn.metrics import accuracy_score
count_vect = CountVectorizer()
print("正確率:", 1-accuracy_score(pre, test_df["tag"]))


# # 預測函數

# In[132]:


def predict(text):
    print(text)
    print(' ')
    docs_news = cutflow(text)
    c=[{"content" : docs_news,
       "tag" : 0}]
    d = pd.DataFrame(c)
    test_counts2 = vec.transform(d["content"])
    pre1 = clf.predict(test_counts2)
    if list(pre1)[0] == '1':
        print("預測的答案:這篇文章偏國民黨")
    if list(pre1)[0] == '0':
        print("預測的答案:這篇文章偏民進黨")


# In[133]:


predict("去你媽的國民黨")


# In[134]:


predict("國民黨好棒棒")


# In[ ]:




