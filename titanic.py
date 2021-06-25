import requests
from csv import writer
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as kme
from sklearn.preprocessing import MinMaxScaler as mms
from collections import Counter

# Task 1
request = requests.get("https://www.imdb.com/title/tt0120338/reviews?ref_=tt_ov_rt")

soup = BeautifulSoup(request.content, 'html.parser')

allReviews = soup.find_all(class_="lister-item-content")

with open('reviews.csv', 'w') as csv_file:
    # write headers for csv file
    csv_writer = writer(csv_file)
    headers = ['Title', 'Description', 'Ratings', 'Helpful']
    csv_writer.writerow(headers)

    # records each review and places into one row of csv file
    for review in allReviews:
        title = review.find(class_="title").get_text().replace('\n', '')
        description = review.find(class_="text show-more__control").get_text().replace('\n', '')
        ratings = review.find('span', attrs={'id': ''}).text.replace('\n', '')
        if (not ratings[0].isdigit()):
            ratings = ""
        else:
            ratings2 = ratings.split("/")
            ratings = ratings2[0]
        helpful = review.find(class_="actions text-muted").get_text().replace('\n', '').strip().split()
        helpful[0] = helpful[0].replace(",", "")
        csv_writer.writerow([title, description, ratings, int(helpful[0])])

# Task 2: k-means clustering
df = pd.read_csv("/Users/mmhki/OneDrive/Documents/CMU/projects/reviews.csv")
# change directory, this one personal

#scaling variables
scalar = mms()
df['Helpful'] = scalar.fit_transform(df[["Helpful"]])
df['Ratings'] = scalar.fit_transform(df[["Ratings"]])

#clustering
km = kme(n_clusters = 3)
df.dropna(subset = ["Ratings"], inplace=True)
yPred = km.fit_predict(df[['Ratings', 'Helpful']])
df['cluster'] = yPred

#plotting
df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0.Ratings, df0['Helpful'], color = 'red')
plt.scatter(df1.Ratings, df1['Helpful'], color = 'green')
plt.scatter(df2.Ratings, df2['Helpful'], color = 'blue')
plt.xlabel("Ratings")
plt.ylabel("Helpful")

plt.show()

# Task 3

# mC = Counter(" ".join(df["Description"]).split()).most_common(10)
# print(mC)
# [('the', 260), ('and', 131), ('of', 121), ('I', 99), ('a', 98), 
# ('to', 89), ('in', 63), ('was', 59), ('it', 56), ('this', 53)]
# most common words in all reviews

# get rid of the common words
mask0 = len(df0["Description"]) * [True]
mask1 = len(df1["Description"]) * [True]
mask2 = len(df2["Description"]) * [True]

for word in ["the", "and", "of", "I", "a", "to", "in", "was", "it", "this"]:
    df0.loc[mask0, "Description"] = df0.loc[mask0, "Description"].str.replace(word, '')
    df1.loc[mask1, "Description"] = df1.loc[mask1, "Description"].str.replace(word, '')
    df2.loc[mask2, "Description"] = df2.loc[mask2, "Description"].str.replace(word, '')

# most counts per word
mC0 = Counter(" ".join(df0["Description"]).split()).most_common(5)
mC1 = Counter(" ".join(df1["Description"]).split()).most_common(5)
mC2 = Counter(" ".join(df2["Description"]).split()).most_common(5)

# print result
print(mC0, mC1, mC2)

# not accurate, too many common words, maybe needs sentiment analysis?
