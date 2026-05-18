"""
Converted from notebook: 1_EDA.ipynb
"""

# # Movie Recommendation for Streaming Platform
# ## Notebook 1 — Exploratory Data Analysis (EDA)
# 
# **Objective:** Understand the MovieLens dataset — its structure, size, users, movies, and rating behaviour — before building any model.
# 
# **Files used:**
# - `ratings.DAT` — user ratings for movies
# - `movies.DAT` — movie titles and genres
# - `users.DAT` — user demographic info

import pandas as pd
import numpy as np

# ## Step 1 — Load the Three Datasets

path        = r'C:\Movie Recommendation project\data\ratings.DAT'
movies_path = r'C:\Movie Recommendation project\data\movies.DAT'
users_path  = r'C:\Movie Recommendation project\data\users.DAT'

ratings = pd.read_csv(path,        sep='::', engine='python',
                      names=['user_id','movie_id','rating','timestamp'], encoding='iso-8859-1')
movies  = pd.read_csv(movies_path, sep='::', engine='python',
                      names=['movie_id','title','genres'],               encoding='iso-8859-1')
users   = pd.read_csv(users_path,  sep='::', engine='python',
                      names=['user_id','gender','age','occupation','zip_code'], encoding='iso-8859-1')

print('Ratings shape :', ratings.shape)
print('Movies shape  :', movies.shape)
print('Users shape   :', users.shape)

# ## Step 2 — Preview Each Dataset

print('=== Ratings ===')
print(ratings.head())
print('\n=== Movies ===')
print(movies.head())
print('\n=== Users ===')
print(users.head())

# ## Step 3 — Map Occupation Codes to Readable Names

occupation_dict = {
    0:'Other', 1:'Academic/Educator', 2:'Artist', 3:'Clerical/Admin',
    4:'College/Grad Student', 5:'Customer Service', 6:'Doctor/Health Care',
    7:'Executive/Managerial', 8:'Farmer', 9:'Homemaker', 10:'K-12 Student',
    11:'Lawyer', 12:'Programmer', 13:'Retired', 14:'Sales/Marketing',
    15:'Scientist', 16:'Self-Employed', 17:'Technician/Engineer',
    18:'Tradesman/Craftsman', 19:'Unemployed', 20:'Writer'
}
users['occupation_name'] = users['occupation'].map(occupation_dict)
users[['user_id','occupation','occupation_name']].head()

# ## Step 4 — Merge All Three Datasets into One

data = pd.merge(ratings, movies, on='movie_id')
data = pd.merge(data,    users,  on='user_id')

print('Merged data shape:', data.shape)
print(data.head())

# ## Step 5 — Basic Data Quality Check

print('=== Missing Values ===')
print(data.isnull().sum())

print('\n=== Data Types ===')
print(data.dtypes)

print('\n=== Rating Range ===')
print(f"Min rating: {data['rating'].min()}   Max rating: {data['rating'].max()}")

# ## Step 6 — Key Dataset Statistics

print(f"Total ratings        : {len(data):,}")
print(f"Unique users         : {data['user_id'].nunique():,}")
print(f"Unique movies        : {data['title'].nunique():,}")
print(f"Avg ratings per user : {data.groupby('user_id')['rating'].count().mean():.1f}")
print(f"Avg ratings per movie: {data.groupby('title')['rating'].count().mean():.1f}")
print(f"Overall avg rating   : {data['rating'].mean():.2f}")

# ## Step 7 — Gender and Age Distribution

print('=== Gender Distribution ===')
print(users['gender'].value_counts())

print('\n=== Age Group Distribution ===')
age_map = {1:'Under 18', 18:'18-24', 25:'25-34', 35:'35-44', 45:'45-49', 50:'50-55', 56:'56+'}
users['age_group'] = users['age'].map(age_map)
print(users['age_group'].value_counts())

# ## Step 8 — Top Occupations Among Users

print('=== Top 10 Occupations ===')
print(users['occupation_name'].value_counts().head(10))

# ## Step 9 — Save Merged Data for Use in Other Notebooks

data.to_csv(r'C:\Movie Recommendation project\data\merged_data.csv', index=False)
print('Saved merged_data.csv')

# ## EDA Summary
# 
# | Observation | Finding |
# |---|---|
# | Dataset size | ~1 million ratings, 6040 users, 3706 movies |
# | Missing values | None (clean dataset) |
# | Rating scale | 1 to 5 |
# | Dominant gender | Male users |
# | Largest age group | 25–34 |
# | Largest occupation | College/Grad Student |
# 
# The data is clean and ready for visualisation and modelling.
