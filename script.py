#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:53:35 2017

@author: shubhabrata
"""

import pandas as pd
from os.path import join
import itertools

data_path = "/media/shubhabrata/DATAPART1/HERE-X/ECMLChallenge"


names_coord = ['coord1', 'coord2']
df_trcoord = pd.read_csv(join(data_path, 'coord_training.txt'), names = names_coord)

feats_ = ['UBlue', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'BI']
dates_ = list(itertools.chain(*[[str(i)]*10 for i in range(1,24)]))
names_features = [i+j for i, j in zip(feats_*23, dates_)]
df_train = pd.read_csv(join(data_path, 'training.txt'), names = names_features)

df_trclass = pd.read_csv(join(data_path, 'training_class.txt'), names = ['class'])