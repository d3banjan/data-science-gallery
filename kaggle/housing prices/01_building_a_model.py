#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:59:30 2018

* Following along with a tutorial to predict housing prices
https://www.kaggle.com/dansbecker/your-first-scikit-learn-model

@author: debanjan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# import data located at
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/downloads/melb_data.csv.zip
data_link = r"melb_data.csv"
melb_data = pd.read_csv(data_link, index_col=0)
# understand the distribution of missing values for later
#msno.matrix(melb_data)
#msno.heatmap(melb_data)

# clean data of all nan/na values
# NOTE this is just so that the regressor doesn't throw any errors
# imputation to be handed later on so as not to miss any valuable data
melb_data = melb_data.dropna(axis=0, how="any")

# TASK from https://www.kaggle.com/dansbecker/selecting-and-filtering-in-pandas
# Print a list of the columns
melb_data.info()
# From the list of columns, find a name of the column with the sales prices of the homes. Use the dot notation to extract this to a variable (as you saw above to create melbourne_price_data.)
melb_prices = melb_data.Price
# Use the head command to print out the top few lines of the variable you just created.
melb_prices.head()
# Pick any two variables and store them to a new DataFrame (as you saw above to create two_columns_of_data.)
# NOTE: I decided to go with more features
important_features = ["Rooms",
                      "Distance",
                      "Bedroom2",
                      "Bathroom",
                      "Car",
                      "Landsize",
                      "YearBuilt"]
melb_X = melb_data[important_features]
# Use the describe command with the DataFrame you just created
# to see summaries of those variables.
melb_X.describe()

# https://www.kaggle.com/dansbecker/your-first-scikit-learn-model
melb_model = DecisionTreeRegressor()
melb_model.fit(melb_X,
               melb_prices)

# some predictions from training data
print("Making predictions for the following 5 houses:")
print(melb_X.head())
print("The predictions are")
print(melb_model.predict(melb_X.head()))
print(melb_prices.head())


# TASK predictions using given columns
data_link = r"train.csv"
iowa_data = pd.read_csv(data_link, index_col=0)
#msno.matrix(iowa_data)
#msno.heatmap(iowa_data)
# data = data.dropna(axis=0, how="any")
# Select the target variable you want to predict. You can go back to the
# list of columns from your earlier commands to recall what it's called
# (hint: you've already worked with this variable). Save this to a new
# variable called y.
iowa_prices = iowa_data.SalePrice
# Create a list of the names of the predictors we will use in the initial
#model. Use just the following columns in the list (you can copy and
# paste the whole list to save some typing, though you'll still need
# to add quotes):
#
# LotArea
# YearBuilt
# 1stFlrSF
# 2ndFlrSF
# FullBath
# BedroomAbvGr
# TotRmsAbvGrd
important_features = ["LotArea",
                      "YearBuilt",
                      "1stFlrSF",
                      "2ndFlrSF",
                      "FullBath",
                      "BedroomAbvGr",
                      "TotRmsAbvGrd"]

# Using the list of variable names you just created, select a new
# DataFrame of the predictors data. Save this with the variable name X.
iowa_X = iowa_data[important_features]

# Create a DecisionTreeRegressorModel and save it to a variable
# (with a name like my_model or iowa_model). Ensure you've done the
# relevant import so you can run this command.
iowa_model = DecisionTreeRegressor()
# Fit the model you have created using the data in X and
# the target data you saved above.
iowa_model.fit(iowa_X,
               iowa_prices)
# Make a few predictions with the model's predict command
# and print out the predictions.
print("Making predictions for the following 5 houses:")
print(iowa_X.head())
print("The predictions are")
print(iowa_model.predict(iowa_X.head()))
print(iowa_prices.head())


# TASK test train split on melbourne data
# NOTE the order of test and training split
train_X,test_X,train_y,test_y = train_test_split(melb_X,melb_prices,random_state=0)
melb_model.fit(train_X,train_y)
predictions = melb_model.predict(test_X)
print(mean_absolute_error(predictions,test_y))

# plot mean absolute error(mae) as a function of tree depth
def get_mae(max_leaf_nodes,
            train_X,
            test_X,
            train_y,
            test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                  random_state=0)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred_y)
    return(mae)

# optimize max_leaf_nodes for melbourne data
train_X,test_X,train_y,test_y = train_test_split(melb_X,melb_prices,random_state=0)
max_nodes_list = list(range(10,700,5))
mae_list =[]
for max_node in max_nodes_list:
    mae = get_mae(max_node,train_X,test_X,train_y,test_y)
    mae_list.append(mae)

# plot max_node vs mae
plt.plot(max_nodes_list,mae_list)
plt.show()
plt.savefig("png/melb_nodes_vs_mae.png")


# optimize max_leaf_nodes for iowa data
train_X,test_X,train_y,test_y = train_test_split(iowa_X,iowa_prices,random_state=0)
max_nodes_list = list(range(10,700,5))
mae_list =[]
for max_node in max_nodes_list:
    mae = get_mae(max_node,train_X,test_X,train_y,test_y)
    mae_list.append(mae)

# plot max_node vs mae
plt.plot(max_nodes_list,mae_list)
plt.savefig("png/iowa_nodes_vs_mae.png")

plt.show()