# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:16:11 2020
@author: Yueran Xu
"""

# %%%% Preliminaries and library loading
import datetime
import os
import pandas as pd
import re
import shelve
import time
import datetime

# libraries to scrape websites
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)
os.chdir('C:\\Users\\Nees-Public\\Desktop\Marketing Analytics\\final')
path_to_driver_3 ='C:/Users/Nees-Public/Desktop/Marketing Analytics/chromedriver_85.exe'
driver         = webdriver.Chrome(executable_path=path_to_driver_3)


# %%%%

# Creating the list of links.
links_to_scrape = ['https://www.amazon.com/Echo-Dot-3rd-Gen-Sandstone/product-reviews/B07PGL2N7J/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=five_star&pageNumber=1',
                   'https://www.amazon.com/Echo-Dot-3rd-Gen-Sandstone/product-reviews/B07PGL2N7J/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=four_star&pageNumber=1',
                   'https://www.amazon.com/Echo-Dot-3rd-Gen-Sandstone/product-reviews/B07PGL2N7J/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=three_star&pageNumber=1',
                   'https://www.amazon.com/Echo-Dot-3rd-Gen-Sandstone/product-reviews/B07PGL2N7J/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=two_star&pageNumber=1',
                   'https://www.amazon.com/Echo-Dot-3rd-Gen-Sandstone/product-reviews/B07PGL2N7J/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=one_star&pageNumber=1']

# Create empty list to store reviews
reviews_all = []

for one_link in links_to_scrape:
    # Finding all the reviews in the website and bringing them to python
    driver.get(one_link)

    # loop over pages until the last page
    condition = True
    count = 0
    while condition:
        # Get list of reviews on that page （10 reviews on a page)
        reviews = driver.find_elements_by_xpath("//div[@class='a-section review aok-relative']")
        r = 0
        # loop over every review on that page
        for r in range(len(reviews)):
            one_review                   = {}
            soup                         = BeautifulSoup(reviews[r].get_attribute('innerHTML'))

            # Get title
            try:
                one_review_title = soup.find('a', attrs={'class':'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'}).text
                                                                
            except:
                one_review_title = soup.find('span', attrs={'class':'a-size-base review-title a-color-base review-title-content a-text-bold'}).text
            one_review['review_title'] = one_review_title.strip('\n')
    
            # Get main text content
            try:
                one_review_text = soup.find('span', attrs={'class':'a-size-base review-text review-text-content'}).text
            except:
                one_review_text = ""
            one_review['review_text'] = one_review_text.strip('\n')[2:]
    
            # Get star rating
            try:
                one_review_stars = re.findall('star-[0-5]',reviews[r].get_attribute('innerHTML'))[0]
                one_review['review_stars'] = ' '.join([one_review_stars.split('-')[1],one_review_stars.split('-')[0]])
            except:
                one_review_stars = ""
                one_review['review_stars'] = ''
            
            # Append data of one review to list
            reviews_all.append(one_review) 
            count += 1
            print(count)
        # Find the 'Next Page' button and go to next page
        try:
            driver.find_element_by_xpath("//li[@class='a-last']").click()
        # If reaches the last page, stop scraping
        except NoSuchElementException:
            condition = False
            
        # Give time to load data
        time.sleep(3)   
# Close driver
driver.close()

# Store data to dataframe, then to csv
reviews_df = pd.DataFrame.from_dict(reviews_all)
reviews_df.to_excel('reviews.xlsx', index=False)

# See summary of dataset
reviews_df.info()
reviews_df.describe()

# %%%% Data Preparation

reviews_df = pd.read_excel('reviews.xlsx')
reviews_df = reviews_df.fillna(' ')
reviews_df['stars'] = reviews_df.review_stars.str.extract('([12345])( star)')[[0]].astype(int)
reviews_df = reviews_df.drop(['review_stars'], axis = 1)

def sentiments(star):
    if (star == 5) or (star == 4):
        return 1
    elif star == 3:
        return 0
    elif (star == 2) or (star == 1):
        return -1

reviews_df['sentiment'] = reviews_df['stars'].apply(sentiments)

# %%%% Snapshot of dataframe - distribution
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

stars, star_counts = np.unique(reviews_df['stars'], return_counts=True)
sentiments, sent_counts = np.unique(reviews_df['sentiment'], return_counts=True)

plt.bar(stars, star_counts, align='center')
plt.xticks(stars)
plt.ylabel('Count')
plt.xlabel('Stars')
plt.title('Reviews Count by Stars')
plt.show()

plt.bar(sentiments, sent_counts, align='center')
plt.xticks(sentiments)
plt.ylabel('Count')
plt.xlabel('Sentiment')
plt.title('Reviews Count by Sentiment')
plt.show()


# %%%% Snapshot of dataframe - wordcloud

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


def plotCloud(text):
    cloud_plot = WordCloud(width = 3000, height = 2000, 
                           background_color='WhiteSmoke', colormap='Set2', 
                           stopwords=stopwords_list).generate(text)
    # Set picture size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(cloud_plot) 
    # No axis details
    plt.axis("off")
    plt.show()

# Create stopword list
stopwords_list = set(stopwords.words('english') + ['-PRON-'])

# Join words for all reviews
all_title = " ".join(review for review in reviews_df.review_title)
all_text = " ".join(review for review in reviews_df.review_text)

plotCloud(all_title)
plotCloud(all_text)

# Split data by sentiment
positive = reviews_df[reviews_df['sentiment'] == 1]
neutural = reviews_df[reviews_df['sentiment'] == 0]
negative = reviews_df[reviews_df['sentiment'] == -1]

# Join words by sentiment
pos_text = " ".join(review for review in positive.review_text)
neu_text = " ".join(review for review in neutural.review_text)
neg_text = " ".join(review for review in negative.review_text)

plotCloud(pos_text)
plotCloud(neu_text)
plotCloud(neg_text)



# %%%% split data into train and test
'''
from sklearn.model_selection import StratifiedShuffleSplit


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in sss.split(reviews_df['review_text'],
                                           reviews_df["sentiment"]): 
    train = reviews_df.reindex(train_index)
    test = reviews_df.reindex(test_index)

print(len(train))
print(train["sentiment"].value_counts()/len(train))
print(len(test))
print(test["sentiment"].value_counts()/len(test))

X_train = train["review_text"]
Y_train = train["sentiment"]
X_test = test["review_text"]
Y_test = test["sentiment"]
'''
# %%%% Split data to train and test set

# Assign random number 
reviews_df['class_num'] = np.random.randn(len(reviews_df.index))
# Split data into 80% training and 20% testing
train = reviews_df[reviews_df['class_num'] <= 0.8]
test = reviews_df[reviews_df['class_num'] > 0.8]

X_train = train["review_text"]
Y_train = train["sentiment"]
X_test = test["review_text"]
Y_test = test["sentiment"]
'''
print(len(train))
print(train["sentiment"].value_counts()/len(train))
print(len(test))
print(test["sentiment"].value_counts()/len(test))
'''
# %%%% create bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report

import numpy as np

# extract features
count_vect = CountVectorizer(lowercase = True, ngram_range = (1,1),
                             max_df = 0.8, min_df = 0.01)
X_train_counts = count_vect.fit_transform(X_train) 

# Build model to simplify process
def modelPredict(model):
    clf = Pipeline([("vect", CountVectorizer()), 
                    ("clf", model)])
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print(str(model))
    print(confusion_matrix(Y_pred, Y_test))
    print(classification_report(Y_pred, Y_test))

modelDict = {'Naive Bayes':MultinomialNB(), 'Logistic Regression':LogisticRegression(),
             'Support Vector Machine':LinearSVC(), 'Decision Tree': DecisionTreeClassifier(),
             'Random Forest':RandomForestClassifier()}

for key, value in modelDict.items():
    print(key)
    modelPredict(value)
  
# %%%% Revised model training excluding neutral sentiment
    
train_new = train[train['sentiment'] != 0]
test_new = test[test['sentiment'] != 0]

X_train_new = train_new["review_text"]
Y_train_new = train_new["sentiment"]
X_test_new = test_new["review_text"]
Y_test_new = test_new["sentiment"]
    
count_vect = CountVectorizer(lowercase = True, ngram_range = (1,1),
                             max_df = 0.8, min_df = 0.01)
X_train_counts_new = count_vect.fit_transform(X_train_new) 
    
def modelPredictNew(model):
    clf = Pipeline([("vect", CountVectorizer()), 
                    ("clf", model)])
    clf.fit(X_train_new, Y_train_new)
    Y_pred_new = clf.predict(X_test_new)
    print(str(model))
    print(confusion_matrix(Y_pred_new, Y_test_new))
    print(classification_report(Y_pred_new, Y_test_new))  
    
for key, value in modelDict.items():
    print(key)
    modelPredictNew(value)   
    
    
    
    
    
'''    
modelPredict(MultinomialNB())
modelPredict(LogisticRegression())
modelPredict(LinearSVC())
modelPredict(DecisionTreeClassifier())
modelPredict(RandomForestClassifier())
modelPredict(MLPClassifier())


# naive bayes
from sklearn.naive_bayes import MultinomialNB
clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), 
                             ("clf_nominalNB", MultinomialNB())])
clf_multiNB_pipe.fit(X_train, Y_train)

predictedMultiNB = clf_multiNB_pipe.predict(X_test)
np.mean(predictedMultiNB == Y_test)

confusion_matrix(predictedMultiNB, Y_test)
print(classification_report(predictedMultiNB, Y_test))

# logistic
from sklearn.linear_model import LogisticRegression
clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), 
                            ("clf_logReg", LogisticRegression())])
clf_logReg_pipe.fit(X_train, Y_train)

predictedLogReg = clf_logReg_pipe.predict(X_test)
np.mean(predictedLogReg == Y_test)

confusion_matrix(predictedLogReg, Y_test)
print(classification_report(predictedLogReg, Y_test))

#SVM
from sklearn.svm import LinearSVC
clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()), 
                               
                               ("clf_linearSVC", LinearSVC())])
clf_linearSVC_pipe.fit(X_train, Y_train)

predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
np.mean(predictedLinearSVC == Y_test)

confusion_matrix(predictedLinearSVC, Y_test)
print(classification_report(predictedLinearSVC, Y_test))

# decision tree
from sklearn.tree import DecisionTreeClassifier
clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()), 
                                  
                                  ("clf_decisionTree", DecisionTreeClassifier())
                                 ])
clf_decisionTree_pipe.fit(X_train, Y_train)

predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
np.mean(predictedDecisionTree == Y_test)

confusion_matrix(predictedDecisionTree, Y_test)
print(classification_report(predictedDecisionTree, Y_test))

#random forest
from sklearn.ensemble import RandomForestClassifier
clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()), 
                                  
                                  ("clf_randomForest", RandomForestClassifier())
                                 ])
clf_randomForest_pipe.fit(X_train, Y_train)

predictedRandomForest = clf_randomForest_pipe.predict(X_test)
np.mean(predictedRandomForest == Y_test)

confusion_matrix(predictedRandomForest, Y_test)
print(classification_report(predictedRandomForest, Y_test))

# neural net
from sklearn.neural_network import MLPClassifier
clf_neuralNets_pipe = Pipeline([("vect", CountVectorizer()), 
                                  
                                  ("clf_neuralNets", MLPClassifier())
                                 ])
clf_neuralNets_pipe.fit(X_train, Y_train)

predictedNeuralNets = clf_neuralNets_pipe.predict(X_test)
np.mean(predictedNeuralNets == Y_test)

confusion_matrix(predictedNeuralNets, Y_test)
print(classification_report(predictedNeuralNets, Y_test))
'''





