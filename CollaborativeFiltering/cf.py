import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

def RMSEandCorrectionRateItem(predictionDF, ratings):
	prediction = []
	actualRating = []

	for line in ratings.itertuples():
		prediction.append(predictionDF.loc[line[2], line[3]])
	prediction = np.array(prediction)
	actualRating = np.array(ratings['stars'])
	#need to ignore the NaN values
	testRMSE = np.sqrt(np.nanmean((prediction - actualRating)**2))
	testSuccessRate = np.mean((prediction >= 4) == (actualRating >= 4))
	return testRMSE, testSuccessRate

def RMSEandCorrectionRateUser(predictionDF, ratings):
	prediction = []
	actualRating = []
	for line in ratings.itertuples():
		if line[2] in predictionDF.index and line[3] in predictionDF.columns:
			prediction.append(predictionDF.loc[line[2], line[3]])
			actualRating.append(line[4])
	prediction = np.array(prediction)
	#need to ignore the NaN values
	testRMSE = np.sqrt(np.nanmean((prediction - actualRating)**2))
	testSuccessRate = np.mean((prediction >= 4) == (actualRating >= 4))
	return testRMSE, testSuccessRate

#return the RMSE to the test data and the item_based CF
def item_based(utilityMatrix, testRatingsFile, trainRatingsFile):

	#first transfer the dataframe to np array of 2D
	matrix = pd.read_csv(utilityMatrix)
	originMatrix = matrix
	matrix.rename(columns={'Unnamed: 0' : 'user_ids'}, inplace=True)# rename the Unnamed: 0 for the user_ids
	users = np.array(matrix[['user_ids']]).flatten()
	#print users

	matrix.drop(['user_ids'], axis=1, inplace=True)
	matrix = np.array(matrix)

	trainRatings = pd.read_csv(trainRatingsFile)
	testRatings = pd.read_csv(testRatingsFile)

	#print matrix
	#print trainRatings
	#print

	numUser = matrix.shape[0]
	numBusiness = matrix.shape[1]
	#userMeans = matrix.mean(axis = 1)
	#print numUser
	#print numBusiness
	#print userMeans

	businessSimilarities = cosine_similarity(matrix.T)
	#check the correctness of similarities
	#print businessSimilarities
	#print businessSimilarities.shape[0]
	#print businessSimilarities.shape[1]

	#make the predictions for currently empty businesses based on the formula of weighted value of similarities
	itemBasedPrediction = np.zeros((numUser, numBusiness))
	for i in range(numUser):
		for j in range(numBusiness):
			activeIndexes = matrix[i, :] != 0
			totalWeight = sum(abs(businessSimilarities[activeIndexes, j]))
			if totalWeight == 0:
				totalWeight = 1.4E-10
			itemBasedPrediction[i, j] = np.dot(matrix[i, :], businessSimilarities[:, j] / totalWeight)

	# add the column name and row name
	itemBasedPredictionDF = pd.DataFrame(itemBasedPrediction, columns = originMatrix.columns, index = users)
	itemBasedPredictionDF[itemBasedPredictionDF < 0] = 0
	itemBasedPredictionDF[itemBasedPredictionDF > 5] = 5

	#prediction for test sets
	testRMSE, testSuccessRate = RMSEandCorrectionRateItem(itemBasedPredictionDF, testRatings)
	trainRMSE, trainSuccessRate = RMSEandCorrectionRateItem(itemBasedPredictionDF, trainRatings)
	return(testRMSE, testSuccessRate, trainRMSE, trainSuccessRate)

def user_based(utilityMatrix, testRatingsFile, trainRatingsFile):
	#first transfer the dataframe to np array of 2D
	#number of users are too big for testing. We user the first 10000 of the users.
	matrix = pd.read_csv(utilityMatrix)
	matrix = matrix.head(10000)
	originMatrix = matrix
	matrix.rename(columns={'Unnamed: 0' : 'user_ids'}, inplace=True)# rename the Unnamed: 0 for the user_ids
	users = np.array(matrix[['user_ids']]).flatten()
	#print users

	matrix.drop(['user_ids'], axis=1, inplace=True)
	matrix[matrix==0] = np.nan
	#for calculating mean
	matrix = np.array(matrix)
	userMeans = np.nanmean(np.where(matrix != 0, matrix, np.nan), 1)
	where_are_Nans = np.isnan(matrix)

	matrix[where_are_Nans] = 0.0
	trainRatings = pd.read_csv(trainRatingsFile)
	testRatings = pd.read_csv(testRatingsFile)

	numUser = matrix.shape[0]
	numBusiness = matrix.shape[1]

	#print numUser
	#print numBusiness
	#print userMeans
	userSimilarities = cosine_similarity(matrix)
	userBasedPrediction = np.zeros((numUser, numBusiness))
	for i in range(numUser):
		for j in range(numBusiness):
			activeIndexes = matrix[:, j] != 0
			totalWeight = sum(abs(userSimilarities[i, activeIndexes]))
			if totalWeight == 0:
				userBasedPrediction[i, j] = userMeans[i]
			else:
				tmp = matrix[:, j]
				normalizedUserVector = tmp - userMeans
				userBasedPrediction[i, j] = userMeans[i] + np.dot(normalizedUserVector, userSimilarities[i, :] / totalWeight)

	userBasedPredictionDF = pd.DataFrame(userBasedPrediction, columns = originMatrix.columns, index = users)
	userBasedPredictionDF[userBasedPrediction < 0] = 0
	userBasedPredictionDF[userBasedPredictionDF > 5] = 5

	testRMSE, testSuccessRate = RMSEandCorrectionRateUser(userBasedPredictionDF, testRatings)
	trainRMSE, trainSuccessRate = RMSEandCorrectionRateUser(userBasedPredictionDF, trainRatings)
	return(testRMSE, testSuccessRate, trainRMSE, trainSuccessRate)

RMSE_item_based = item_based("trainMatrix.csv", "testCityRate.csv", "trainCityRate.csv")
print RMSE_item_based
RMSE_user_based = user_based("trainMatrix.csv", "testCityRate.csv", "trainCityRate.csv")
print RMSE_user_based