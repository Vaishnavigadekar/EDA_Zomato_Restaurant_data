#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS ON ZOMATO_RESTAURANT_DATA

# # Introduction

#  Exploratory Data Analysis (EDA) is a preliminary step of Machine Learning and is used extensively in this field. Although it is not necessary to perform EDA to build models, but it is definitely recommended as it helps to know the data better. If performed correctly, it gives us insights which are not easy to witness directly.In this notebook, I have performed a detailed analysis on Indian Restaurants Dataset from Zomato(link). This notebook can be used as a manual to perform basic to intermediate EDA on any dataset. Following are the things that you will learn from this project :-
# - Knowing basic composition of data
# 
# - Removing duplicates
# 
# - Dealing with missing values
# 
# - Understanding features
# 
# - Plotting horizontal bar charts (multicolor)
# 
# - Using groupby, apply, and unique functions
# 
# - Scatter plot
# 
# - Box plot
# 
# - Density plot
# 
# - Bar Charts
# 
# - Drawing insights and conclusions from data

# # Project Outline

# - Importing data set
# 
# -   Preprocessing
# - - Exploring data
# 
# - - Removing duplicates
# 
# - - Dealing with missing values
# 
# - - Omitting not useful features
# 
# - EDA
# 
# -  Restaurant Chains
# 
# - -  Chains vs Outlets
# 
# - -  Top Restaurant Chains (by number of outlets)
# 
# - - Top Restaurant Chains (by average ratings)
# 
# -  Establishment Types
# 
# - - Number of Restaurants
# 
# - -   Average Rating, Votes, and Photo count
# 
# - Cities
# 
# - - Number of Restaurants
# - - - Average Rating, Votes, and Photo count
# - - - Cuisine
# - - Total number of unique cuisines
# - - - Number of Restaurants
# - - - Highest rated cuisines
# - - Highlights
# - - - Number of Restaurants
# - - - Highest rated features
# - - - Highlights wordcloud
# 
# - Rating and cost
# 
# - - Rating Distribution
# - - - Average Cost for two distribution
# - - - Price range count
# - - - Relation between Average price for two and Rating
# - - - Relation between Price Range and Rating
# - - - Relation between Votes and Rating
# 
# - Conclusions

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import random 


# In[75]:


#need to treat duplicate data
#duplicate data is one which your id is repeated more than twice.


# # Preprocessing 

# ## Exploring data

# In[76]:


data = pd.read_csv("zomato_restaurants_in_India.csv")
data


# In[77]:


data.head()


# In[78]:


data.tail()


# In[79]:


#getting data of any random rows
data.sample(3)


# In[80]:


data.columns


# In[81]:


#we have 99 unique cities
data.city.unique()


# In[82]:


data[data["city"]=="Pune"]


# In[83]:


data[data["city"]=="Nashik"]


# In[84]:


data.shape


# In[85]:


#info() is used to know about the count,null and data type properties.
data.info()


# # Removing duplicates

# ### It is important to remove duplicate rows to avoid biasness in our analysis.Since res_id is unique identifier of our restaurant ,we can use it to remove duplicates

# In[86]:


data.drop_duplicates(["res_id"],keep ='first',inplace=True)
data.shape


# ### Around 70% of data has duplicate rows.therefore its good Practice that we removed it before starting with the analysis and visualization. 

# # Dealing with missing values

# In[87]:


data.isnull().sum()


# ### We can see that 5 variables has missing values.Since zipcode has ~80% of missing data,its better not to consider it at all.The other four features can be dealt with certain manupulation as they do not contain maximum missing values. 

# # Omitting not useful features

# ### Here we will look at each feature and decide to consider them for our  analysis or not :-
# 
# - res_id - Unique ID for each restaurant.
# 
# - name - Name is useful since we will use it to find top restaurants.
# 
# - establishment - let's see what type of values we have in establishment.

# In[88]:


data["establishment"].unique()


# In[89]:


print(data["establishment"].unique()[0])
print(type(data["establishment"].unique()[0]))


# ### Establishment looks like a useful feature to perform EDA , however each value has an unwanted square brackets and quotes which seems noisy . Let's remove them with apply() function. Also, we have one value which is empty string ,lets rename it as NA to avoid confussion.

# In[90]:


#Removing [ ' '] from each value
print(data["establishment"].unique()[0])
data["establishment"] = data["establishment"].apply(lambda x:x[2:-2])
print(data["establishment"].unique()[0])

#changing ' ' to 'NA'
print(data["establishment"].unique())
data["establishment"] = data["establishment"].apply(lambda x:np.where(x=="","NA",x))
print(data["establishment"].unique())


# - url - URL is the link to restaurant's page which is not usefull for us.
# 
# - address - not useful since it has strings and its difficult to classify.
# 
# - city - Lets check unique cities.

# In[91]:


len(data[data["city"]=="Shimla"])


# In[92]:


len(data[data["city"]=="Nashik"])


# In[93]:


len(data["city"].unique())


# In[94]:


data["city"].unique()


# - locality - Lets see number of unique values for locality.

# In[95]:


data["locality"].nunique()


# ### Although locality is interesting feature,but since this feature has so many unique classes ,we will avoid it.
# 
# - latitude - Can be helpful while using geographic maps.
# 
# - longitude - same as above.
# 
# - zipcode - it contains 80% of missing values.
# 
# - country_id - Since the dataset is for Indian restaurants, there should be just one unique id here.lets check.
# 
# - locality_verbose - contains lot of unique values same as of locality.

# In[96]:


data["country_id"].unique()


# In[97]:


data["locality_verbose"].nunique()


# ### Cuisines - This feature has missing values . although it has 9382 unique classes ,we can see that each restaurant has a list of cuisines and the composition is the reason why we have so many different cuisines classes .lets check actual number of unique classes .but first we will replace the missing values with labels.

# In[98]:


print(data["cuisines"].nunique())
print(data["cuisines"].unique())


# In[99]:


data["cuisines"] = data["cuisines"].fillna("No Cuisines")


# In[100]:


cuisines = []
data["cuisines"].apply(lambda x : cuisines.extend(x.split(", ")))
cuisines = pd.Series(cuisines)
print("Total number of unique cuisines = " , cuisines.nunique())


# ### Timings - this also has missing data,however it has 7740 unique classes.Also ,it is not structured even if we try to reduce the number clases like we did in cuisines . its better to omit it altogether.

# In[101]:


print(data["timings"].nunique())
print(data["timings"].unique())


# In[102]:


sns.scatterplot(data = data , x ='average_cost_for_two' , y = 'res_id')


# ### average_cost_for_two - this is the interesting feature for our project,although it contains the value "0" but ww will keep it as outlier.

# In[103]:


data["average_cost_for_two"].nunique()


# In[104]:


data["price_range"].unique()


# In[105]:


#only one class not useful
data["currency"].unique()


# ### highlights - They represents certain features that the restuarant specializes in and want to highlight to their costomers. Each restuarant has a list of highlights which makes the composition diffirent for each one.We will filter this and find out total unique highlights from all restaurants.

# In[106]:


print(data["highlights"].nunique())
print(data["highlights"].unique())


# In[107]:


h1 = []
data["highlights"].apply(lambda x : h1.extend(x[2:-2].split("', '")))
h1 = pd.Series(h1)
print("Total number of unique highlights = " , h1.nunique())


# - aggregate_rating - Rating given to the restaurant.
# 
# - rating_text - Chracteristics of numeric rating into bins by using labels.We will be using direct rating in our analysis so we will be ignoring this.
# 
# - votes - Number of votes contributing to the rating 
# 
# - photo_counts - photo uploaded in reviews.
# 
# let's check the range and mean for above features.

# In[108]:


data[["aggregate_rating","votes","photo_count"]].describe().loc[["mean","min","max"]]


# ### Rating ranges between 0 and 5 while 42539 are the maximun votes given to a restaurant . the negative value in votes must be an outlier.

# - opentable_support - Not useful since no restaurant has True value for this.
# 
# - delivery - This feature has 3 classes but there is no explaination for these classes .we can consider -1 and 0 to be one class or ignore this feature for now.
# 
# - takeaway - Again not useful since it has only one class.

# In[109]:


data["opentable_support"].unique()


# In[110]:


data["delivery"].unique()


# In[111]:


data["takeaway"].unique()


# # Exploratory Data Analysis(EDA)

# ## Restaurant chains
# 
# Here chain represents restuarants with more than one outlet

# ### chains Vs outlets

# In[112]:


outlets = data["name"].value_counts()
outlets


# In[113]:


chains = outlets[outlets >= 2]
single = outlets[outlets == 1]


# In[114]:


data.shape


# In[115]:


chains


# In[116]:


print("Total Restaurants = " , data.shape[0])
print("Total Restaurant that are part of some chain = " , data.shape[0] - single.shape[0] )
print("percentage of Restaurants that are part of chain = " , np.round((data.shape[0] - single.shape[0])/data.shape[0],2)*100 , "%")


# ### 35% of total restaurants are part of some kind of resturant chain.Here,we should account for cases where two diffirent restuarants might have exact same name but not related to each other.

# ### Top Restaurant chains (by number of outlets)
# 
# let's plot the horizontal bar graph to look at the top 10 resturants chains .

# In[117]:


top10_chains = data["name"].value_counts()[:10].sort_values(ascending = True)


# In[118]:


single


# In[119]:


height = top10_chains.values
bars = top10_chains.index
y_pos = np.arange(len(bars))
fig = plt.figure(figsize=[11,7],frameon = False)
ax = fig.gca()
ax.spines["top"].set_visible('#424242')
ax.spines["right"].set_visible(False)
ax.spines['left'].set_color("#424242")
ax.spines['bottom'].set_color('#424242')
plt.barh(y_pos, height )
plt.xticks(color = '#424242')
plt.yticks(y_pos,bars,color='#424242')
plt.xlabel("Number of outlets in India")


plt.title("Top 10 Restaurant chain in India(by Number of outlets)")
plt.show()


# This chart is majorly dominaed by big fast food chains Top restaurant chains (by average rating)
# Here we will look at top chains by their ratings. I have set the criteria of number of outlets to greater than 4 to remove some outliers.

# In[120]:


outlets = data["name"].value_counts()


# In[121]:


atleast_5_outlets = outlets[outlets[outlets>4]]


# In[122]:


top10_chains2 = data[data["name"].isin(atleast_5_outlets.index)].groupby("name").mean()["aggregate_rating"].sort_values(ascending = False)[:10].sort_values(ascending = False)


# In[123]:


height = pd.Series(top10_chains2.values).map(lambda x : np.round(x, 2))
bars = top10_chains2.index
y_pos = np.arange(len(bars))

fig = plt.figure(figsize=[11,7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible("#424242")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ['#fded86', '#fce36b', '#f7c65d', '#f1a84f', '#ec8c41', '#e76f34', '#e25328', '#b04829', '#7e3e2b', '#4c3430']
plt.barh(y_pos, height, color=colors)

plt.xlim(3)
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets in India")

for i, v in enumerate(height):
    ax.text(v + 0.01, i, str(v), color='#424242')
plt.title("Top 10 Restaurant chain in India (by average Rating)")

plt.show()


# Interestingly, no fast food chain appears in this chart. To maintain a high rating, restaurants needs to provide superior service which becomes impossible with booming fast food restaurant in every street

# ## Establishment Types

# ### Number of restaurants (by establishment type)

# In[124]:


est_count = data.groupby("establishment").count()["res_id"].sort_values(ascending=False)[:5]

fig = plt.figure(figsize=[8,5], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ["#2d0f41",'#933291',"#e65586","#f2a49f","#f9cdac"]
plt.bar(est_count.index, est_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 25000, 5000), color="#424242")
plt.xlabel("Top 5 establishment types")

for i, v in enumerate(est_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by establishment type)")

plt.show()


# ### Top 3 represents more casual and quick service restaurants, then from 4-6 we have dessert based shops.

# ### Average rating, votes and photos (by Establishment) Here, we will not plot each graph since it will make this notebook filled with horizontal bar charts. I see horizontal bar charts the only option to display results of this kind when we have lots of classes to compare (here 10 classes). Let's look at value_counts( ) directly

# In[125]:


rating_by_est = data.groupby("establishment").mean()["aggregate_rating"].sort_values(ascending=False)[:10]
rating_by_est


# In[126]:


data.groupby("establishment").mean()["votes"].sort_values(ascending=False)[:10]


# In[127]:


data.groupby("establishment").mean()["photo_count"].sort_values(ascending=False)[:10]


# It can be concluded that establishments with alcohol availability have highest average ratings, votes and photo uploads.

# # Cities

# ### Number of restaurants (by city)

# In[128]:


city_counts = data.groupby("city").count()["res_id"].sort_values(ascending=True)[-10:]

height = pd.Series(city_counts.values)
bars = city_counts.index
y_pos = np.arange(len(bars))

fig = plt.figure(figsize=[11,7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible("#424242")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ['#dcecc9', '#aadacc', '#78c6d0', '#48b3d3', '#3e94c0', '#3474ac', '#2a5599', '#203686', '#18216b', '#11174b']
plt.barh(y_pos, height, color=colors)

plt.xlim(3)
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets")

for i, v in enumerate(height):
    ax.text(v + 20, i, str(v), color='#424242')
plt.title("Number of restaurants (by city)")


plt.show()


# ## As expected, metro cities have more number of restaurants than others with South India dominating the Top 4.

# # Average rating, votes and photos (by city) 

# In[129]:


rating_by_city = data.groupby("city").mean()["aggregate_rating"].sort_values(ascending=False)[:10]
rating_by_city


# In[130]:


data.groupby("city").mean()["votes"].sort_values(ascending=False)[:10]


# In[131]:


data.groupby("city").mean()["photo_count"].sort_values(ascending=False)[:10]


# ## Gurgaon has highest rated restaurants whereas Hyderabad has more number of critics. Mumbai and New Delhi dominates for most photo uploads per outlet.

# # Cuisine

# ## Unique Cuisine

# In[132]:


print("Total number of unique cuisines = ", cuisines.nunique())


# ## Number of restaurants (by cuisine)

# In[133]:


c_count = cuisines.value_counts()[:5]

fig = plt.figure(figsize=[8,5], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ['#4c3430', '#b04829', '#ec8c41', '#f7c65d','#fded86']
plt.bar(c_count.index, c_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 30000, 5000), color="#424242")
plt.xlabel("Top 5 cuisines")

for i, v in enumerate(c_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by cuisine type)")


plt.show()


# ## Surprisingly, Chinese food comes second in the list of cuisines that Indians prefer, even more than fast food, desserts and South Indian food.

# # Highest rated cuisines

# In[134]:


data["cuisines2"] = data['cuisines'].apply(lambda x : x.split(", "))

cuisines_list = cuisines.unique().tolist()
zeros = np.zeros(shape=(len(cuisines_list),2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum","Total"])


# In[135]:


for i, x in data.iterrows():
    for j in x["cuisines2"]:
        c_and_r.loc[j]["Sum"] += x["aggregate_rating"]  
        c_and_r.loc[j]["Total"] += 1


# In[136]:


c_and_r["Mean"] = c_and_r["Sum"] / c_and_r["Total"]
c_and_r


# In[137]:


c_and_r[["Mean","Total"]].sort_values(by="Mean", ascending=False)[:10]


# ## We can ignore a few cuisines in this list since they are available in less number. But the overall conclusion which can be drawn is that International (and rarely available) cuisines are rated higher than local cuisines.

# # Highlights/Features of restaurants

# In[139]:


print("Total number of unique cuisines = ", h1.nunique())


# ## Number of restaurants (by highlights)

# In[140]:


h_count = h1.value_counts()[:5]

fig = plt.figure(figsize=[10,6], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ['#11174b', '#2a5599', '#3e94c0', '#78c6d0', '#dcecc9']
plt.bar(h_count.index, h_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 70000, 10000), color="#424242")
plt.xlabel("Top 5 highlights")

for i, v in enumerate(h_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by highlights)")


plt.show()


# In[142]:


h_count = h1.value_counts()[:5]

fig = plt.figure(figsize=[10,6], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")
colors = ['#11174b', '#2a5599', '#3e94c0', '#78c6d0', '#dcecc9']
plt.bar(h_count.index, h_count.values, color=colors)

plt.xticks(range(0, 6), color="#424242")
plt.yticks(range(0, 70000, 10000), color="#424242")
plt.xlabel("Top 5 highlights")

for i, v in enumerate(h_count):
    ax.text(i-0.2, v+500, str(v), color='#424242')
plt.title("Number of restaurants (by highlights)")


plt.show()


# # Highest rated highlights

# In[143]:


data["highlights"][0]


# In[145]:


data["highlights2"] = data['highlights'].apply(lambda x : x[2:-2].split("', '"))

hl_list = h1.unique().tolist()
zeros = np.zeros(shape=(len(hl_list),2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum","Total"])


# In[146]:


for i, x in data.iterrows():
    for j in x["highlights2"]:
        h_and_r.loc[j]["Sum"] += x["aggregate_rating"]  
        h_and_r.loc[j]["Total"] += 1


# In[147]:


h_and_r["Mean"] = h_and_r["Sum"] / h_and_r["Total"]
h_and_r


# ## We can safely ignore highlights which have a frequency of less than 10 since they can be considered as outliers. Features like Gastro pub, Craft beer, Romantic dining and Sneakpeek are well received among customers.

# # Relation between Average price for two and Rating

# In[148]:


plt.plot("average_cost_for_two","aggregate_rating", data=data, linestyle="none", marker="o")
plt.xlim([0,6000])
plt.title("Relationship between Average cost and Rating")
plt.xlabel("Average cost for two")
plt.ylabel("Ratings")
plt.show()


# ## There is definetely a direct relation between the two. Let's take a smaller sample to draw a clearer scatter plot.

# In[149]:


plt.plot("average_cost_for_two","aggregate_rating", data=data.sample(1000), linestyle="none", marker="o")
plt.xlim([0,3000])
plt.show()


# ## This relation concludes that that as average cost for two increases, there is a better chance that the restaurant will be rated highly. Let's look at price range for a better comparison.

# ## Relation between Price range and Rating

# In[150]:


np.round(data[["price_range","aggregate_rating"]].corr()["price_range"][1],2)


# In[151]:


sns.boxplot(x='price_range', y='aggregate_rating', data=data)
plt.ylim(1)
plt.title("Relationship between Price range and Ratings")
plt.show()


# ## Now, it is clear. The higher the price a restaurant charges, more services they provide and hence more chances of getting good ratings from their customers.

# # Conclusions

# - Approx. 35% of restaurants in India are part of some chain.
# 
# - Domino's Pizza, Cafe Coffee Day, KFC are the biggest fast food chains in the country with most number of outlets.
# 
# - Barbecues and Grill food chains have highest average ratings than other type of restaurants.
# 
# - Quick bites and casual dining type of establishment have most number of outlets.
# 
# - Establishments with alcohol availability have highest average ratings, votes and photo uploads.
# 
# - Banglore has most number of restaurants.
# 
# - Gurgaon has highest rated restaurants (average 3.83) whereas Hyderabad has more number of critics (votes). 
# 
# - Mumbai and New -Delhi dominates for most photo uploads per outlet.
# 
# - After North Indian, Chinese is the most prefered cuisine in India.
# 
# - International cuisines are better rated than local cuisines.
# 
# - Gastro pub, Romantic Dining and Craft Beer features are well rated by customers.
# 
# - Majority of restaurants are budget friendly with average cost of two between Rs.250 to Rs.800.
# 
# - There are less number of restaurants at higher price ranges.
# 
# - As the average cost of two increases, the chance of a restaurant having higher rating increases.
# 
# Now we have come to the end of this project, I hope you learned some new tricks.

# In[ ]:




