# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:24:50 2014
Reduces the dataset from 20GB to about 1GB (349.655.789 lines to 15.349.956 lines) or "the category subset"
@author: sroy
"""

from datetime import datetime

loc_offers = "/home/sroy/Desktop/KaggleDataSet/ValueShoppers/offers.csv"
loc_transactions = "/home/sroy/Desktop/KaggleDataSet/ValueShoppers/transactions.csv"
loc_reduced = "/home/sroy/Desktop/KaggleDataSet/ValueShoppers/reduced2.csv" # will be created

def reduce_data(loc_offers, loc_transactions, loc_reduced):

  start = datetime.now()
  #get all categories on offer in a dict
  offers = {}
  for e, line in enumerate( open(loc_offers) ):
    offers[ line.split(",")[1] ] = 1
  #open output file
  with open(loc_reduced, "wb") as outfile:
    #go through transactions file and reduce
    reduced = 0
    for e, line in enumerate( open(loc_transactions) ):
      if e == 0:
        outfile.write( line ) #print header
      else:
        #only write when category in offers dict
          if line.split(",")[3] in offers:
            outfile.write( line )
            reduced += 1
      #progress
      if e % 5000000 == 0:
        print e, reduced, datetime.now() - start
  print e, reduced, datetime.now() - start

reduce_data(loc_offers, loc_transactions, loc_reduced)
