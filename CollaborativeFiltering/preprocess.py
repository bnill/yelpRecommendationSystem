import os
import numpy as np
import pandas as pd
from pandas import DataFrame

currDir = os.path.dirname(os.path.abspath(__file__))
review = pd.read_json(currDir + "/data/yelp_academic_dataset_review.json", lines = True)
business = pd.read_json(currDir + "/data/yelp_academic_dataset_business.json", lines = True)
user = pd.read_json(currDir + "/data/yelp_academic_dataset_user.json", lines = True)
#review = pd.read_json(currDir + "/data/review_test.json", lines = True)
#business = pd.read_json(currDir + "/data/business_test.json", lines = True)
#user = pd.read_json(currDir + "/data/user_test.json", lines = True)

#set the city to one of the 9 cities given in data
cityName = "Las Vegas"
trainRatio = 0.9
businessFilterNumReviews = 50
userFilterNumReviews = 50

cityBusiness = business[business.city == cityName]

#drop the users and business with less than 50 reviews
userFiltered = user[user.review_count > userFilterNumReviews]
cityBusinessFiltered = cityBusiness[cityBusiness.review_count > businessFilterNumReviews]
print "initial filter completed"
print userFiltered.shape
print cityBusinessFiltered.shape
print business.shape

# merge review, user and business
review_business_user = pd.merge(pd.merge(review, cityBusinessFiltered, on="business_id"), userFiltered, on="user_id")
#col_drop = review_business_user.colomns.difference(["business_id", "user_id", "stars"])
#review_business_user.drop(col_drop, axis = 1, inplace = True)
print review_business_user
review_business_user = review_business_user[["business_id", "user_id", "stars_x"]]
print "merge complete"

#omit the floating part
review_business_user.stars_x = review_business_user.stars_x.astype(int)
res = pd.pivot_table(review_business_user, index="business_id", columns="user_id", values="stars_x")

res.to_csv('ratings.csv')
print "User-Business-Rating Matrix completed"


