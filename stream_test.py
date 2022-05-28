#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import streamlit as st

reddit_df = pd.read_csv(
    'C:/Users/user/Downloads/Reddit Cryptocurrency Daily Discussion Comments + Sentiment Labels 28-05-2022.csv'
)
scaler = StandardScaler()
label2int = {'Neutral': 0, 'Bullish': 1, 'Bearish': -1, 'Spam': 0}
testtest_df2 = reddit_df.copy()
testtest_df2 = pd.DataFrame(
    [label2int[label] for label in testtest_df2['Sentiment Label']])
testtest_df2 = pd.concat([reddit_df, testtest_df2], axis=1)
plot_df = testtest_df2[::-1]
plot_df = plot_df.reset_index(drop=True)
plot_df['Index'] = range(1, len(plot_df) + 1)
values = plot_df[0].tolist()
xx = pd.DataFrame({'Overall Sentiment': values},
                  index=plot_df['Created (US EDT Time)'])
xx = xx.reset_index()
xx['Created (US EDT Time)'] = pd.to_datetime(xx['Created (US EDT Time)'])
xx = xx.resample('H', on='Created (US EDT Time)').sum()
xx = xx.reset_index()
xx[['Overall Sentiment']] = scaler.fit_transform(xx[['Overall Sentiment']])

bar_fig = px.bar(xx,
                 x='Created (US EDT Time)',
                 y='Overall Sentiment',
                 color='Overall Sentiment',
                 title='Overall Sentiment Per Hour (US EDT Time)')

line_fig = px.line(xx,
                   x="Created (US EDT Time)",
                   y="Overall Sentiment",
                   title='Overall Sentiment Per Hour (US EDT Time)')

# Plot!
st.plotly_chart(bar_fig, use_container_width=True)
st.plotly_chart(line_fig, use_container_width=True)

