# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:56:36 2014
# Reads the csv files and train the model
@author: sroy
"""

import pandas as pd
import numpy as np
import scipy as sp
#import scipy.interpolate
#from scipy import signal

import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import datetime as dt
import json
import sys
from os import listdir, getcwd
from os.path import isfile, join, abspath

#from DataPreparationFunctions import *
#from io_functions import *

def prepare_data(obj, pars): 
    
    CustomerID = obj['CustomerID']
    StoreID = obj['StoreID']
    OfferID = obj['OfferID']
    CompanyID = obj['CompanyID']
    BrandID =  obj['BrandID']
    MarketID = obj['MarketID']
    
    
    
    
    
    # Run the code
if __name__ == '__main__':    
    t1 = dt.datetime.now()
    
    offer_data = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/ValueShoppers/offers.csv', sep=',')
    train_data = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/ValueShoppers/trainHistory.csv', sep=',')
    transaction_data = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/ValueShoppers/reduced2.csv', sep=',') 
    obj = {}
    obj['CustomerID'] = id
    obj['StoreID'] = chain
    obj['OfferID'] = offer
    obj['CompanyID'] = company
    obj['BrandID'] = brand    
    obj['MarketID'] = market # Geographic region
    obj['NumRepeatTrips'] = repeattrips
    obj['IsRepeat'] = repeater
    obj['OfferDate'] = offerdate
    obj['PurchaseDate'] = date # The date of purchase
    obj['ProductType'] = dept # An aggregate grouping of the Category (e.g. water)
    obj['ProductSubType'] =  category # The product category (e.g. sparkling water)
    obj['PurchaseAmount'] = productsize # The amount of the product purchase (e.g. 16 oz of water)
    obj['ProductUnit'] = productmeasure # The units of the product purchase (e.g. ounces)
    obj['PurchaseNumUnit'] = purchasequantity # The number of units purchased
    obj['PurchaseAmount'] = purchaseamount # The dollar amount of the purchase
    obj['Quantity'] = quantity # The number of units one must purchase to get the discount
    obj['OfferValue'] = offervalue # The dollar value of the offer
    list_of_objects.append(obj)
    
    t2 = dt.datetime.now()
    print 'Total data preparation time: ', (t2-t1).seconds , 'sec'
