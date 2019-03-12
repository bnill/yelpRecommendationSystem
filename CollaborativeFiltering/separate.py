import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import random

currDir = os.path.dirname(os.path.abspath(__file__))
review = pd.read_json(currDir + "/data/yelp_academic_dataset_review.json", lines = True)
business = pd.read_json(currDir + "/data/yelp_academic_dataset_business.json", lines = True)
user = pd.read_json(currDir + "/data/yelp_academic_dataset_user.json", lines = True)

#review = pd.read_json(currDir + "/data/review_test.json", lines = True)
#business = pd.read_json(currDir + "/data/business_test.json", lines = True)
#user = pd.read_json(currDir + "/data/user_test.json", lines = True)

#set the city to one of the 9 cities given in data
cityName = "Las Vegas"
trainRatio = 0.7
businessFilterNumReviews = 500
userFilterNumReviews = 50

#get city business Ids
cityBusiness = business[business.city == cityName]
cityBusinessIds = cityBusiness[cityBusiness.review_count >= businessFilterNumReviews].business_id
#print "cityBusinessId"
#print cityBusinessIds

#Use cityBusinessIds to get the reviewIds
cityBusinessSet = set(cityBusinessIds)
cityReviews = review[review.business_id.isin(cityBusinessSet)]
#print "cityReviews"
#print cityReviews

#get Users
userFiltered = user[user.review_count >= userFilterNumReviews]
#include review_count here for getting average
userFiltered = userFiltered[['user_id', 'review_count']]
cityReview_user = pd.merge(cityReviews, userFiltered, on="user_id")
userSize = len(set(userFiltered.user_id))
businessSize = len(set(cityBusinessIds))
#the upper bound of user and city business
print userSize
print businessSize
print cityReview_user.shape[0]#2D Array

#start of separating the data
# we are going to have a np.array as indexes for the train data set, the others for the test data set
total = cityReview_user.shape[0]
tmpPool = [True, False]
trainIndexes = np.random.choice(tmpPool, total, p=[trainRatio, 1 - trainRatio])
#print trainIndexes
testIndexes = np.logical_not(trainIndexes)
#print testIndexes
cityRate = cityReview_user[['user_id', 'business_id', 'stars']]
trainCityRate = cityRate[trainIndexes]
testCityRate = cityRate[testIndexes]

#calculating the proportion of train
userList = trainCityRate['user_id'].unique()
userNum = len(userList)
cityBusinessList = trainCityRate['business_id'].unique()
cityBusinessNum = len(cityBusinessList)
print "count of user is %d" %(userNum)
print "count of business is %d" %(cityBusinessNum)

#separation of test, get the data and user both exists in train set
testCityRate = testCityRate.loc[testCityRate['user_id'].isin(userList),:]
testCityRate = testCityRate.loc[testCityRate['business_id'].isin(cityBusinessList),:]
ratio = float(testCityRate.shape[0]) / (testCityRate.shape[0] + trainCityRate.shape[0]) if (testCityRate.shape[0] + trainCityRate.shape[0]) > 0 else 1.0
print "The portion of test data set is %f" %(ratio)

#all User and Business List
allUserList = cityRate['user_id'].unique()

#create training matrix
trainArray = np.zeros((userNum, cityBusinessNum))
trainMatrix = DataFrame(trainArray, columns = cityBusinessList, index = userList)
for line in trainCityRate.itertuples():
	trainMatrix.loc[line[1], line[2]] = line[3]

trainMatrix.to_csv("trainMatrix.csv")
print "trainMatrix saved"
testCityRate.to_csv("testCityRate.csv")
print "testCityRate saved"
trainCityRate.to_csv("trainCityRate.csv")
print "trainCityRate saved"


