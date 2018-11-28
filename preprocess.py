# python 3

import pandas as pd
import numpy as np
import sys
import random as rd
from sklearn.decomposition import PCA

FILE_REVIEWS = "data/train_reviews.csv"
FILE_USERS = "data/users.csv"
FILE_BUSINESSES = "data/business.csv"
FILE_OUT_ROOT = "data/pp"

DROP_REVIEWS = ["text", "date"]
DROP_USERS = ["name", "friends"]
DROP_BUSINESSES = ["address", "attributes", "name", "postal_code", "hours", "hours_Friday", "hours_Monday", 
								 "hours_Saturday", "hours_Sunday", "hours_Thursday", "hours_Tuesday", "hours_Wednesday"]

# "numerical" includes one-hot categorical
NUMERICAL = [
							# reviews
							"cool_review", "funny_review", "stars_review", "useful_review",
							
							# users
							"average_stars", "compliment_cool", "compliment_cute", 
							"compliment_funny", "compliment_hot", "compliment_list", 
							"compliment_more", "compliment_note", "compliment_photos", 
							"compliment_plain", "compliment_profile", "compliment_writer", 
							"cool_user", "elite", "fans", "funny_user", "review_count", "useful_user", 
							
							# businesses
							"attributes_AcceptsInsurance", 
							"attributes_BYOB", "attributes_BikeParking", "attributes_BusinessAcceptsBitcoin", 
							"attributes_BusinessAcceptsCreditCards", "attributes_ByAppointmentOnly", 
							"attributes_Caters", "attributes_CoatCheck", "attributes_Corkage", 
							"attributes_DogsAllowed", "attributes_DriveThru", 
							"attributes_GoodForDancing", "attributes_GoodForKids", 
							"attributes_HappyHour", "attributes_HasTV", "attributes_Open24Hours", 
							"attributes_OutdoorSeating", "attributes_RestaurantsCounterService", 
							"attributes_RestaurantsDelivery", "attributes_RestaurantsGoodForGroups", 
							"attributes_RestaurantsPriceRange2", # 1~4
							"attributes_RestaurantsReservations", "attributes_RestaurantsTableService", 
							"attributes_RestaurantsTakeOut", "attributes_WheelchairAccessible", 
							"is_open", "latitude", "longitude", "review_count", "stars_business"
						]
DATE = ["date", "yelping_since"]
# format: { "average" / "loud" / "soft" }
CATEGORICAL = [	"attributes_AgesAllowed", "attributes_Alcohol", "attributes_BYOBCorkage", 
								"attributes_NoiseLevel", "attributes_RestaurantsAttire", "attributes_Smoking", 
								"attributes_WiFi", "city", "neighborhood", "state"]
# format: "Cajun/Creole, Southern, Restaurants"
CATEGORICAL_LIST = ["categories"]
# format: "{'romantic': False, 'intimate': True, 'casual': False}"
CATEGORICAL_ONEHOT = ["attributes_Ambience", "attributes_BestNights", 
											"attributes_BusinessParking", "attributes_DietaryRestrictions", 
											"attributes_GoodForMeal", "attributes_HairSpecializesIn", "attributes_Music"]

class Preprocess(object):
	def __init__(self):
		self.reviews = pd.read_csv(FILE_REVIEWS, index_col="review_id")
		self.users = pd.read_csv(FILE_USERS, index_col="user_id")
		self.businesses = pd.read_csv(FILE_BUSINESSES, index_col="business_id")
		self.data = []

	def drop_cols(self):
		self.reviews.drop(DROP_REVIEWS, axis="columns")
		self.users.drop(DROP_USERS, axis="columns")
		self.businesses.drop(DROP_BUSINESSES, axis="columns")

	def sample(self, frac_sample, frac_seed):
		self.reviews = self.reviews.sample(frac=frac_sample, random_state=frac_seed)
		self.users = self.users.sample(frac=frac_sample, random_state=frac_seed)
		self.businesses = self.businesses.sample(frac=frac_sample, random_state=frac_seed)

	def combine_data(self):
		self.data = self.reviews
		self.data = self.data.join(self.users, on="user_id", lsuffix="_review", rsuffix="_user")
		self.data = self.data.join(self.businesses, on="business_id", lsuffix="_review", rsuffix="_business")

	def transform(self):
		# [date] -> timestamp (num)
		# elite -> one-hot (None / Some)
		# categorical -> one-hot
		# categorical_list -> one-hot
		# categorical_onehot -> one-hot (unpack)
		for e in DATE:
			self.data[e] = pd.to_datetime(self.data[e], format="%Y-%m-%d")
			print(e)
			print(self.data[e])
		# self.data.elite = 

	# def impute(self):

	# def pca(self):

	# def clean(self):

	# def write_csv(self):


if __name__ == "__main__":
	pp = Preprocess()
	pp.drop_cols() # drop irrelevant features
	pp.sample(frac_sample=0.05, frac_seed=1) # sample a portion of the data for testing
	pp.combine_data() # pull in user and business data into each review
	pp.data.describe().to_csv("out2")
	pp.data.describe(include="all").to_csv("out3")
	pp.transform() # transform numerical to normalized (?), categorical to one-hot, dates to numerical timestamps
	# pp.impute() # Impute means for numeric data, "missing" for categorical data
	# pp.pca() # principal-component analysis, prune 80% least irrelevant features
	# pp.clean() # cleanup
	# pp.write_csv() # write to file
