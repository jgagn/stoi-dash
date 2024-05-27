#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:23:04 2024

@author: joelgagnon
"""

#%% Description

# New file to create the athlete database from a single master csv file

#%% Import Libraries

import pandas as pd
import numpy as np
import pickle
import chardet #to detect character encoding - making sure accents and such are saved properly in names

#%% Import CSV Data

#file path and csv file name
path = "test_data/Combined"
csv_file = "gymcanmag.csv"

# Detect the encoding
with open(path+"/"+csv_file, 'rb') as f:
    result = chardet.detect(f.read())

# Extract the detected encoding
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

# Read the CSV file with the detected encoding
database = pd.read_csv(path+"/"+csv_file, encoding=encoding)

# print(database.head())

#%% Acronyms

competition_acronyms = {"EC2024": "Elite Canada 2024",
                 "CC2023": "Canadian Championships 2023",
                 "CC2024": "Canadian Championships 2024"
                    }

category_acronyms = {"SR21":"Senior (21+)",
                     "SRNG": "Senior (NextGen)",
                     "JR17": "Junior (17-18)",
                     "JR15": "Junior (15-16)",
                     "Aspire": "Aspire",
                     }

#%% Rename column headers
# the format is a bit silly, where there are two levels of headers
# the key is that the data goes D score, Score and Rank
order = ["D","Score","Rk"]
Ename = "E"
# apparatus = ["Floor","Pommel Horse","Rings","Vault","Parallel Bars","High Bar","AllAround"]

#create a dictionary where the csv appartus names are keys to desired apparatus abbreviation values
#two-letter acronyms we want to use
tlas = ["FX","PH","SR","VT","PB","HB","AA"]

# abbrev_dict = {apparatus[0]:tlas[0],
#                apparatus[1]:tlas[1],
#                apparatus[2]:tlas[2],
#                apparatus[3]:tlas[3],
#                apparatus[4]:tlas[4],
#                apparatus[5]:tlas[5],
#                apparatus[6]:tlas[6],
#                }
new_columns = []
i = 0
for col in database.columns:
    #check if the column is a D, Score or Rank
    # print(f"i = {i}")
    if (order[0] in col):
        new_col = tlas[i]
        new_columns.append(new_col+"_"+order[0])
    elif (order[1] in col):
        new_col = tlas[i]
        new_columns.append(new_col+"_"+order[1])
    elif (order[2] in col):
        new_col = tlas[i]
        new_columns.append(new_col+"_"+order[2])
        #increment i as we've gone through all orders
        i+=1
    elif "Unnamed" in col:
        #if the column contains "Unnamed" we should skip it, otherwise use the original column
        pass
    else:
        new_columns.append(col)

#apply new column names
database.columns = new_columns

#%% Organize into athlete specific database

athlete_database = {}

#get all athlete names
# Extract unique values from the column
athletes = database['Athlete'].unique()

#need to do a bit of prep work to figure out which categories and competitiond days occured for all competitions
comp_overview={}

#extract unique competitions
competitions = database['Competition']

#for each competition, extract categories
for comp in competitions:
    #lets add the competition to the overview
    comp_overview[comp] = {}
    
    # Step 1: Filter the DataFrame
    competition_df = database[database['Competition'] == comp]
    
    # Step 2: Extract unique values from the 'values' column of the filtered DataFrame
    categories = competition_df['Category'].unique()
    
    # Step 3: For all categories, get unique competition day results
    for category in categories:
        
        category_df = database[competition_df['Category'] == category]
        results = category_df['Results'].unique()
        
        #lets add the category to the comp overview
        comp_overview[comp][category] = results
    

#lets add the comp overivew data to the athlete_database

athlete_database['overview'] = comp_overview

#add acronym data here
athlete_database['competition_acronyms'] = competition_acronyms
athlete_database['category_acronyms'] = category_acronyms

for athlete in athletes:
    #create an dictionary entry for the athlete in the athlete_database
    athlete_database[athlete] = {}
    #Lets start by going through each competition
    for comp in comp_overview.keys():
        #let's see if the athlete competed in that comp
        
        # Filter the DataFrame
        matching_entries = database[(database['Athlete'] == athlete) & (database['Competition'] == comp)]
        # Check if there is any matching entry
        if not matching_entries.empty:
            #1. create a dictionary entry for this competition
            athlete_database[athlete][comp] = {}
            
            
            #2. let's obtain what category they were in for this competition
            filtered_df = database[(database['Athlete'] == athlete) & (database['Competition'] == comp)]
            category = filtered_df.iloc[0]['Category']
            
            #3. append the category they are in for this competition -> important as athletes may change categories
            athlete_database[athlete][comp]['category'] = category
            
            #4. Get what days they would've competed at this comp based on their category
            days = comp_overview[comp][category]

            #5. Loop through these competition days, and append all data to athlete_database if they exist
            for day in days:
                #check if it exists
                filtered_df = database[(database['Athlete'] == athlete) & (database['Competition'] == comp) & (database['Results'] == day)]
                #if its not empty, lets append the data
                if not filtered_df.empty:
                    # print(f"{athlete} competeted {day} at {comp}")
                    athlete_database[athlete][comp][day] = {}
                    # print(filtered_df)
                    #Now append all data
                    for tla in tlas:
                        #query the dataframe to obtain all data
                        athlete_database[athlete][comp][day][tla] = {}
                        for value in order:
                            val = filtered_df[f'{tla}_{value}']
                            try:
                                athlete_database[athlete][comp][day][tla][value] = float(val.iloc[0])
                            except:
                                #I want to put a nan if its not floatable
                                athlete_database[athlete][comp][day][tla][value] = np.nan

                        #Data does't have E score - doing math
                        D = athlete_database[athlete][comp][day][tla][order[0]]
                        Score = athlete_database[athlete][comp][day][tla][order[1]]
                        #Score is D + E
                        
                        try:
                            E = float(Score) - float(D)
                            #print(f"Score: {float(Score)}, D: {float(D)}")
                        except:
                            E = np.nan
                        #print(E)
                        try:
                            athlete_database[athlete][comp][day][tla][Ename] = float(E)
                        except:
                            athlete_database[athlete][comp][day][tla][Ename] = str(E)
        else:
            #they did not compete in this competion
            print(f"{athlete} did not compete at {comp}")
         
        

#%% I want to pickle my database 

# File path to save the pickled database
database_filename = "gymcan_mag_athletes"
file_path = path+"/"+database_filename

# Pickle the database
with open(file_path, 'wb') as f:
    pickle.dump(athlete_database, f)

print("Database pickled successfully.")
