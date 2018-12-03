# python 3

import pandas as pd
import numpy as np
import sys
import random as rd
import ast
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from joblib import dump, load

FILE_REVIEWS = "data/train_reviews.csv"
FILE_USERS = "data/users.csv"
FILE_BUSINESSES = "data/business.csv"
FILE_TEST = "data/test_queries.csv"
FILE_OUT_ROOT = "data/pp"

DROP_REVIEWS = ["cool", "date", "funny", "text", "useful"] # drop all since we don't have this info in the test data
DROP_USERS = ["name"]
DROP_BUSINESSES = ["address", "attributes", "name", "postal_code", "hours", "hours_Friday", "hours_Monday", 
								 "hours_Saturday", "hours_Sunday", "hours_Thursday", "hours_Tuesday", "hours_Wednesday"]

# "numerical" includes one-hot categorical
NUMERICAL = [
							
							# users
							"average_stars", "compliment_cool", "compliment_cute", 
							"compliment_funny", "compliment_hot", "compliment_list", 
							"compliment_more", "compliment_note", "compliment_photos", 
							"compliment_plain", "compliment_profile", "compliment_writer", 
							"cool", "elite", "fans", "funny", "review_count_review", "useful", 
							
							# businesses
							"attributes_RestaurantsPriceRange2", # 1~4
							"is_open", "latitude", "longitude", "review_count_business", "stars" ]
DATE = ["yelping_since"]
# format: { "average" / "loud" / "soft" }
CATEGORICAL = [	"attributes_AgesAllowed", "attributes_Alcohol", "attributes_BYOBCorkage", 
								"attributes_NoiseLevel", "attributes_RestaurantsAttire", "attributes_Smoking", 
								"attributes_WiFi", "city", "neighborhood", "state",

								# binary with missing values: treat as categorical on True / False / None
								"attributes_AcceptsInsurance", 
								"attributes_BYOB", "attributes_BikeParking", "attributes_BusinessAcceptsBitcoin", 
								"attributes_BusinessAcceptsCreditCards", "attributes_ByAppointmentOnly", 
								"attributes_Caters", "attributes_CoatCheck", "attributes_Corkage", 
								"attributes_DogsAllowed", "attributes_DriveThru", 
								"attributes_GoodForDancing", "attributes_GoodForKids", 
								"attributes_HappyHour", "attributes_HasTV", "attributes_Open24Hours", 
								"attributes_OutdoorSeating", "attributes_RestaurantsCounterService", 
								"attributes_RestaurantsDelivery", "attributes_RestaurantsGoodForGroups", 
								"attributes_RestaurantsReservations", "attributes_RestaurantsTableService", 
								"attributes_RestaurantsTakeOut", "attributes_WheelchairAccessible" ]
# format: "Cajun/Creole, Southern, Restaurants"
CATEGORICAL_LIST = ["categories"]
# format: "{'romantic': False, 'intimate': True, 'casual': False}"
CATEGORICAL_ONEHOT = ["attributes_Ambience", "attributes_BestNights", 
											"attributes_BusinessParking", "attributes_DietaryRestrictions", 
											"attributes_GoodForMeal", "attributes_HairSpecializesIn", "attributes_Music" ]

class Preprocess(object):
	def __init__(self):
		self.reviews = pd.read_csv(FILE_REVIEWS, index_col="review_id")
		self.users = pd.read_csv(FILE_USERS, index_col="user_id")
		self.businesses = pd.read_csv(FILE_BUSINESSES, index_col="business_id")
		self.y = self.reviews['stars']
		self.reviews = self.reviews.drop('stars', axis='columns')
		self.X = []
		self.pca = None
		self.X_pca = []

	def drop_cols(self):
		self.reviews = self.reviews.drop(DROP_REVIEWS, axis="columns")
		self.users = self.users.drop(DROP_USERS, axis="columns")
		self.businesses = self.businesses.drop(DROP_BUSINESSES, axis="columns")

	def sample(self, frac_sample, frac_seed):
		self.reviews = self.reviews.sample(frac=frac_sample, random_state=frac_seed)

	def combine_data(self):
		self.X = self.reviews
		self.X = self.X.join(self.users, on="user_id", lsuffix="_review", rsuffix="_user")
		self.X = self.X.join(self.businesses, on="business_id", lsuffix="_review", rsuffix="_business")
		self.X = self.X.drop(["user_id", "business_id"], axis="columns")

	def transform(self):
		# elite -> count # years elite
		self.X.elite = self.X.elite.apply(lambda x: 0 if x == 'None' else len(x.split(',')))

		# friends -> count # friends
		self.X.friends = self.X.friends.apply(lambda x: 0 if x == 'None' else len(x.split(',')))

		# date -> timestamp
		for attr in DATE:
			self.X[attr] = pd.to_datetime(self.X[attr], format="%Y-%m-%d").apply(lambda x: x.timestamp())

		# categorical -> one-hot
		for attr in CATEGORICAL:
			for attr_val in self.X[attr].unique():
				if attr_val != attr_val: # NaN
					attr_val = 'None'
				data_vals = self.X[attr].apply(lambda x: 1 if x == attr_val else 0)
				# print(attr + '_' + attr_val)
				self.X.insert(len(self.X.columns), attr + '_' + str(attr_val), data_vals)
			self.X = self.X.drop(attr, axis='columns')
		# print(self.X)

		# categorical_list (comma-separated) -> one-hot
		for attr in CATEGORICAL_LIST:
			attr_vals = set()
			for s in self.X[attr]:
				if s != s: continue # NaN
				attr_vals |= set(s.split(', '))
			for attr_val in attr_vals:
				data_vals = self.X[attr].apply(lambda x: 1 if x == x and attr_val in set(x.split(', ')) else 0)
				self.X.insert(len(self.X.columns), attr + '_' + attr_val, data_vals)
			self.X = self.X.drop(attr, axis='columns')

		# categorical_onehot (json string) -> one-hot (unpack)
		for attr in CATEGORICAL_ONEHOT:
			attr_vals = set()
			for s in self.X[attr]:
				if s != s: continue # NaN
				attr_vals |= set(ast.literal_eval(s).keys())
			for attr_val in attr_vals:
				data_vals = self.X[attr].apply(lambda x: 
					1 if (x == x and 
								attr_val in ast.literal_eval(x) and 
								ast.literal_eval(x)[attr_val] == True) 
								else 0)
				self.X.insert(len(self.X.columns), attr + '_' + attr_val, data_vals)
			self.X = self.X.drop(attr, axis='columns')

		# finally, transform all fields to float
		self.X = self.X.apply(lambda x: x.apply(float))

	def impute_numerical(self):
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		self.X[NUMERICAL] = imp.fit_transform(self.X[NUMERICAL])

	def normalize(self):
		self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
		self.X = self.X.fillna(0)

	def pca_fit(self, n_attrs):
		pca = PCA(n_components=n_attrs, svd_solver='auto', random_state=1)
		data = pca.fit(np.array(self.X))
		self.pca = pca
		# print(data.explained_variance_ratio_)
		# print(sum(data.explained_variance_ratio_))

	def pca_transform(self):
		columns = ['pca_%i' % i for i in range(self.pca.n_components)]
		self.X_pca = pd.DataFrame(self.pca.transform(self.X), columns=columns, index=self.X.index)

	def export(self):
		self.X.to_csv("data/pp_train_X_raw.csv")
		self.X_pca.to_csv("data/pp_train_X_pca.csv")
		self.y.to_csv("data/pp_train_y.csv", header=['stars'])
		dump(self.pca, 'data/pca_dump.joblib')





class Preprocess_test(object):
	def __init__(self):
		self.X = pd.read_csv(FILE_TEST)
		self.users = pd.read_csv(FILE_USERS, index_col="user_id")
		self.businesses = pd.read_csv(FILE_BUSINESSES, index_col="business_id")
		self.pca = load('data/pca_dump.joblib')
		self.X_pca = []

	def drop_cols(self):
		self.users = self.users.drop(DROP_USERS, axis="columns")
		self.businesses = self.businesses.drop(DROP_BUSINESSES, axis="columns")

	def sample(self, frac_sample, frac_seed):
		self.X = self.X.sample(frac=frac_sample, random_state=frac_seed)

	def combine_data(self):
		self.X = self.X.join(self.users, on="user_id", lsuffix="_review", rsuffix="_user")
		self.X = self.X.join(self.businesses, on="business_id", lsuffix="_review", rsuffix="_business")
		self.X = self.X.drop(["user_id", "business_id"], axis="columns")

	def transform(self):
		# elite -> count # years elite
		self.X.elite = self.X.elite.apply(lambda x: 0 if x == 'None' else len(x.split(',')))

		# friends -> count # friends
		self.X.friends = self.X.friends.apply(lambda x: 0 if x == 'None' else len(x.split(',')))

		# date -> timestamp
		for attr in DATE:
			self.X[attr] = pd.to_datetime(self.X[attr], format="%Y-%m-%d").apply(lambda x: x.timestamp())

		# categorical -> one-hot
		for attr in CATEGORICAL:
			for attr_val in self.X[attr].unique():
				if attr_val != attr_val: # NaN
					attr_val = 'None'
				data_vals = self.X[attr].apply(lambda x: 1 if x == attr_val else 0)
				# print(attr + '_' + attr_val)
				self.X.insert(len(self.X.columns), attr + '_' + str(attr_val), data_vals)
			self.X = self.X.drop(attr, axis='columns')
		# print(self.X)

		# categorical_list (comma-separated) -> one-hot
		for attr in CATEGORICAL_LIST:
			attr_vals = set()
			for s in self.X[attr]:
				if s != s: continue # NaN
				attr_vals |= set(s.split(', '))
			for attr_val in attr_vals:
				data_vals = self.X[attr].apply(lambda x: 1 if x == x and attr_val in set(x.split(', ')) else 0)
				self.X.insert(len(self.X.columns), attr + '_' + attr_val, data_vals)
			self.X = self.X.drop(attr, axis='columns')

		# categorical_onehot (json string) -> one-hot (unpack)
		for attr in CATEGORICAL_ONEHOT:
			attr_vals = set()
			for s in self.X[attr]:
				if s != s: continue # NaN
				attr_vals |= set(ast.literal_eval(s).keys())
			for attr_val in attr_vals:
				data_vals = self.X[attr].apply(lambda x: 
					1 if (x == x and 
								attr_val in ast.literal_eval(x) and 
								ast.literal_eval(x)[attr_val] == True) 
								else 0)
				self.X.insert(len(self.X.columns), attr + '_' + attr_val, data_vals)
			self.X = self.X.drop(attr, axis='columns')

		# finally, transform all fields to float
		self.X = self.X.apply(lambda x: x.apply(float))

	def fit_to_train_dimensions(self):
		train_X_columns = pd.read_csv('data/pp_train_X_raw.csv', index_col='review_id').columns
		# print(train_X_columns)
		# remove extras
		cols = [col for col in self.X.columns if col in set(train_X_columns)]
		self.X = self.X[cols]
		# add missing
		for col in train_X_columns:
			if col not in self.X:
				self.X[col] = 0 # ?

	def impute_numerical(self):
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		self.X[NUMERICAL] = imp.fit_transform(self.X[NUMERICAL])

	def normalize(self):
		self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
		self.X = self.X.fillna(0)

	def pca_transform(self):
		columns = ['pca_%i' % i for i in range(self.pca.n_components)]
		self.X_pca = pd.DataFrame(self.pca.transform(self.X), columns=columns, index=self.X.index)

	def export(self):
		self.X.to_csv("data/pp_test_X_raw.csv")
		self.X_pca.to_csv("data/pp_test_X_pca.csv")

if __name__ == "__main__":
	pp_train = Preprocess()
	pp_train.drop_cols() # drop irrelevant features
	pp_train.sample(frac_sample=0.001, frac_seed=1) # sample a portion of the data for testing
	pp_train.combine_data() # pull in user and business data into each review
	pp_train.transform() # transform numerical to normalized (?), categorical to one-hot, dates to numerical timestamps
	pp_train.impute_numerical() # Impute means for numerical data
	pp_train.normalize()
	pp_train.pca_fit(20) # principal-component analysis, get top n highest-variance features
	pp_train.pca_transform()
	pp_train.export() # write to file

	pp_test = Preprocess_test()
	pp_test.drop_cols() # drop irrelevant features
	# pp_test.sample(frac_sample=0.001, frac_seed=1) # sample a portion of the data for testing
	pp_test.combine_data() # pull in user and business data into each review
	pp_test.transform() # transform numerical to normalized (?), categorical to one-hot, dates to numerical timestamps
	pp_test.fit_to_train_dimensions() # add missing attrs, drop unsees attrs
	pp_test.impute_numerical() # Impute means for numerical data
	pp_test.normalize()
	pp_test.pca_transform()
	pp_test.export() # write to file
