#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[13]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[14]:


df.head()


# In[9]:


df.shape


# In[15]:


df['user_id']


# In[16]:


df['user_id'].nunique()


# In[17]:


df['item_id'].nunique()


# In[24]:


movies_title=pd.read_csv(r'u.item',sep="\|", header = None, encoding= 'ISO-8859-1')


# In[25]:


movies_title.shape


# In[30]:


movies_title=movies_title[[0,1]]
movies_title.columns=["item_id","title"]
movies_title.head()


# In[32]:


df=pd.merge(df,movies_title,on="item_id")


# In[33]:


df


# In[34]:


df.tail()


# In[35]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[36]:


ratings.head()


# In[38]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# In[39]:


df.head()


# In[40]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[41]:


moviemat.head()


# In[46]:


youngguns_user_ratings=moviemat['Young Guns (1988)']


# In[47]:


youngguns_user_ratings.head(10)


# In[49]:


similar_to_youngguns=moviemat.corrwith(youngguns_user_ratings)


# In[50]:


similar_to_youngguns


# In[51]:


corr_youngguns=pd.DataFrame(similar_to_youngguns,columns=['correlation'])


# In[54]:


corr_youngguns.dropna(inplace=True)


# In[55]:


corr_youngguns


# In[56]:


corr_youngguns.head()


# In[60]:


corr_youngguns.sort_values('correlation',ascending=False).head(10)


# In[61]:


ratings


# In[62]:


corr_youngguns=corr_youngguns.join(ratings['num of ratings'])


# In[63]:


corr_youngguns


# In[64]:


corr_youngguns.head()


# In[68]:


corr_youngguns[corr_youngguns['num of ratings']>100].sort_values('correlation',ascending=False)


# In[98]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[99]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[100]:


predict_my_movie.head()


# In[ ]:




