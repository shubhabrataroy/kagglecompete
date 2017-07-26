#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:26:45 2017

@author: shubhabrataroy
"""

import os
from email.parser import Parser
import pandas as pd
from os.path import join

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.
    From = []
    To = []
    Cc = []
    Bcc = []
    Body = []
    Date = []
    Subject = []

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            #print root
            filepathAbs = os.path.join(root, filename)
            filepathTrimmed = os.path.join(root[60:], filename)
            fp = open(filepathAbs, 'r').read()
            email = Parser().parsestr(fp)
            file_paths.append(filepathTrimmed)  # Add it to the list.
            From.append(email['from'])
            To.append(email['to'])
            Cc.append(email['cc'])
            Bcc.append(email['bcc'])
            Subject.append(email['subject'])
            Date.append(email['Date'])
            s = email.get_payload(decode=True)
            msg = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            Body.append(msg)

    return file_paths, From, To, Cc, Bcc, Body, Date, Subject  

# Run the above function and store its results in variables.  
path_ = "/Users/shubhabrataroy/Desktop/Freelance/PhilipMoris/maildir" 
file_paths, From, To, Cc, Bcc, Body, Date, Subject  = get_filepaths(path_)

df = pd.DataFrame({'path': file_paths, 'from': From, 'to': To, 'cc': Cc, 'bcc': Bcc, 'body': Body, 'date': Date, 'subject': Subject})
df.to_csv(join(path_, 'emails.csv'), index = 'False')