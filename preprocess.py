# python 3

import pandas as pd
import numpy as np
import sys
import random as rd # for sampling
import ast # parse shitty object-like strings in business.csv
import pathlib # mkdir

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load # save and load the pca object

FILE_TRAIN_REVIEWS = "data/train_reviews.csv"
FILE_QUERY_REVIEWS = "data/test_queries.csv"
FILE_USERS = "data/users.csv"
FILE_BUSINESSES = "data/business.csv"
FILE_OUT_ROOT = "data_pp/"

DROP_REVIEWS = ["cool", "date", "funny", "text", "useful"] # drop all since we don't have this info in the query data
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
	def __init__(self, reviews, raw_y=None, pca=None, scaler=None):
		self.reviews = reviews
		self.users = pd.read_csv(FILE_USERS, index_col="user_id")
		self.businesses = pd.read_csv(FILE_BUSINESSES, index_col="business_id")
		self.pca = pca
		self.scaler = scaler
		self.raw_y = raw_y

		self.X = []
		self.X_pca = []


	def drop_cols(self):
		self.reviews = self.reviews.drop(DROP_REVIEWS, axis="columns", errors='ignore')
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

	def fit_cols(self, train_X_cols):
		# print(train_X_cols)
		# remove extras
		cols = [col for col in self.X.columns if col in set(train_X_cols)]
		self.X = self.X[cols]
		# add missing
		for col in train_X_cols:
			if col not in self.X:
				self.X[col] = 0 # ?

	def impute_numerical(self):
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		self.X[NUMERICAL] = imp.fit_transform(self.X[NUMERICAL])

	def normalize(self):
		self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
		self.X = self.X.fillna(0)

	def standardize(self):
		if self.scaler == None:
			self.scaler = StandardScaler()
			self.scaler.fit(self.X)
		# [print(e) for e in self.X.axes[1]]
		# self.X = pd.DataFrame(data=self.scaler.fit_transform(self.X), index=self.X.columns)
		self.X.update(self.scaler.transform(self.X))
		# [print(e) for e in self.X.axes[1]]
		self.X = self.X.fillna(0)

	def pca_fit(self, n_attrs):
		self.pca = PCA(n_components=n_attrs, svd_solver='auto', random_state=1)
		data = self.pca.fit(np.array(self.X))
		# print(data.explained_variance_ratio_)
		# print(sum(data.explained_variance_ratio_))

	def pca_transform(self):
		columns = ['pca_%i' % i for i in range(self.pca.n_components)]
		self.X_pca = pd.DataFrame(self.pca.transform(self.X), columns=columns, index=self.X.index)

	def export(self, prefix, dump_y=False, dump_objs=False):
		self.X.to_csv(FILE_OUT_ROOT + "pp_" + prefix + "_X_raw.csv", index=False)
		self.X_pca.to_csv(FILE_OUT_ROOT + "pp_" + prefix + "_X_pca.csv", index=False)
		if dump_y:
			y = self.X.index
			y = y.to_frame().join(self.raw_y)['stars']
			y.to_csv(FILE_OUT_ROOT + "pp_" + prefix + "_y.csv", index=False, header=True)
		if dump_objs:
			dump(self.scaler, FILE_OUT_ROOT + 'scaler_dump.joblib')
			dump(self.pca, FILE_OUT_ROOT + 'pca_dump.joblib')


if __name__ == "__main__":

	# make directory for preprocessed data
	pathlib.Path(FILE_OUT_ROOT).mkdir(parents=True, exist_ok=True)

	# generate pca for training data
	print('on training data...')
	train_reviews_X = pd.read_csv(FILE_TRAIN_REVIEWS, index_col="review_id")
	train_y = train_reviews_X['stars']
	train_reviews_X = train_reviews_X.drop('stars', axis='columns')
	pp_train = Preprocess(train_reviews_X, raw_y=train_y)
	print('drop_cols...')
	pp_train.drop_cols() # drop irrelevant features
	print('sample...')
	pp_train.sample(frac_sample=0.001, frac_seed=1) # sample a portion of the data (for dev purposes)
	print('combine_data...')
	pp_train.combine_data() # pull in user and business data into each review
	print('transform...')
	pp_train.transform() # transform numerical to normalized (?), categorical to one-hot, dates to numerical timestamps
	print('impute_numerical...')
	pp_train.impute_numerical() # Impute means for numerical data
	print('standardize...')
	pp_train.standardize()
	print('pca_fit...')
	pp_train.pca_fit(20) # principal-component analysis, get top n highest-variance features
	print('pca_transform...')
	pp_train.pca_transform()
	print('export...')
	pp_train.export(prefix='train', dump_y=True, dump_objs=True) # write to file

	# generate pca for test_queries data
	print()
	print('on query data...')
	query_reviews_X = pd.read_csv(FILE_QUERY_REVIEWS, index_col=None)
	query_reviews_X.index.name = 'id'
	pp_query = Preprocess(query_reviews_X, pca=pp_train.pca, scaler=pp_train.scaler)
	print('drop_cols...')
	pp_query.drop_cols() # drop irrelevant features
	print('combine_data...')
	pp_query.combine_data() # pull in user and business data into each review
	print('transform...')
	pp_query.transform() # transform numerical to normalized (?), categorical to one-hot, dates to numerical timestamps
	print('fit_cols...')
	pp_query.fit_cols(pp_train.X.columns) # add missing attrs, drop unseen attrs, wrt trained cols
	print('impute_numerical...')
	pp_query.impute_numerical() # Impute means for numerical data
	print('standardize...')
	pp_query.standardize()
	print('pca_transform...')
	pp_query.pca_transform()
	print('export...')
	pp_query.export(prefix='query') # write to file
