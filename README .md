# Movie Recommendation System for Streaming Platform

> Recommending a movie is easy but recommending the right movie  one a specific person will actually enjoy, based on the patterns in everything they have watched before  is a fundamentally different problem.
This project builds a personalized recommendation engine that learns from user rating history using Collaborative Filtering and SVD-based Matrix Factorization. Instead of suggesting what is popular, it predicts what a particular user would rate highly even for films they have never seen  by decomposing the entire user-movie rating matrix and finding hidden patterns in taste. Model quality is measured with RMSE and MAE to ensure the predictions are not just plausible but genuinely accurate.

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Dataset Information](#dataset-information)
- [Methods](#methods)
- [Project Workflow](#project-workflow)
- [Project Structure](#project-structure)
- [Notebook Descriptions](#notebook-descriptions)
- [Feature Engineering](#feature-engineering)
- [Handling Data Sparsity](#handling-data-sparsity)
- [Machine Learning Methods](#machine-learning-methods)
- [Model Evaluation](#model-evaluation)
- [How to Run This Project](#how-to-run-this-project)
- [Future Improvements](#future-improvements)
- [Key Learnings](#key-learnings)
- [Author](#author)

---

## Overview

This project is an end-to-end data analytics and machine learning system designed to recommend movies to users of a streaming platform based on their past ratings and behaviour. The project covers the complete pipeline including:

- Exploratory Data Analysis (EDA)
- Data Visualisation and Insight Generation
- User-Based Collaborative Filtering using Cosine Similarity
- SVD-based Matrix Factorisation for improved recommendations
- Model Evaluation using RMSE, MAE, Precision@K and Recall@K

The dataset used is the MovieLens 1M dataset — a well-known benchmark dataset for recommender systems research containing approximately 1 million ratings from 6,040 users across 3,706 movies.

---

## Project Objectives

- Build a personalised movie recommendation engine using Collaborative Filtering
- Explore and visualise user behaviour, rating patterns, and genre trends from the MovieLens dataset
- Upgrade the baseline cosine similarity approach using SVD-based Matrix Factorisation
- Evaluate recommendation quality using standard ML evaluation metrics
- Understand the tradeoffs between Memory-Based and Model-Based Collaborative Filtering

---

## Technologies Used

### Programming Language
- Python 3.x

### Libraries and Frameworks

| Library | Purpose |
|---|---|
| Pandas | Data loading, merging, and manipulation |
| NumPy | Matrix operations and numerical computation |
| Matplotlib | Visualisation and plotting |
| Seaborn | Statistical visualisation |
| Scikit-learn | Cosine similarity, train/test split, RMSE, MAE |
| SciPy | SVD decomposition (`scipy.sparse.linalg.svds`) |
| Jupyter Notebook | Interactive development environment |

---

## Dataset Information

Dataset: **MovieLens 1M**

Three source files are used:

| File | Contents |
|---|---|
| `ratings.DAT` | User ratings for movies (user_id, movie_id, rating, timestamp) |
| `movies.DAT` | Movie titles and genre labels (movie_id, title, genres) |
| `users.DAT` | User demographic information (user_id, gender, age, occupation, zip_code) |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total ratings | ~1,000,000 |
| Unique users | 6,040 |
| Unique movies | 3,706 |
| Rating scale | 1 to 5 |
| Missing values | None |

### Target Variable

The recommendation goal is to predict which unrated movies a given user is most likely to enjoy, based on the implicit patterns in their existing ratings.

---

## Methods

1. **Data Ingestion** — Loaded all three DAT files using Pandas with the `::` separator and latin encoding.
2. **Data Merging** — Merged ratings, movies, and users into a single unified dataframe on `movie_id` and `user_id`.
3. **Exploratory Data Analysis** — Analysed rating distributions, user activity, genre frequency, gender and age group patterns.
4. **Visualisation** — Generated 11 charts covering rating distribution, top-rated movies, genre analysis, demographic breakdowns, and user activity.
5. **Feature Engineering** — Mapped occupation codes to readable names; mapped age codes to age group labels for cleaner analysis.
6. **Collaborative Filtering (User-Based):**
   - Built a user-movie pivot matrix
   - Computed cosine similarity between all user pairs
   - Recommended movies using weighted average of top-5 similar users (threshold ≥ 0.3)
   - Filtered out already-watched movies
7. **SVD — Matrix Factorisation:**
   - Split data into 80% train and 20% test using stratified random split
   - Built user-movie matrix from training data only
   - Applied mean-centring to remove user rating bias
   - Decomposed the matrix using `scipy.sparse.linalg.svds` with k=50 latent factors
   - Reconstructed predicted ratings for all user-movie pairs
8. **Model Evaluation:**
   - RMSE and MAE for rating prediction accuracy
   - Precision@K and Recall@K for ranking quality
   - Side-by-side comparison of Cosine Similarity CF vs SVD

---

## Project Workflow

```text
Raw Data (ratings.DAT, movies.DAT, users.DAT)
          ↓
   Data Loading & Merging
          ↓
Exploratory Data Analysis (EDA)
          ↓
     Visualisation
          ↓
User-Based Collaborative Filtering
    (Cosine Similarity)
          ↓
  SVD Matrix Factorisation
          ↓
   Model Evaluation
  (RMSE, MAE, Precision@K, Recall@K)
```

---

## Project Structure

```text
Movie-Recommendation-System/
│
├── data/
│   ├── ratings.DAT
│   ├── movies.DAT
│   ├── users.DAT
│   └── merged_data.csv              ← generated by Notebook 1
│
├── notebooks/
│   ├── 1_EDA.ipynb                  ← Data loading, merging, EDA
│   ├── 2_Visualization.ipynb        ← 11 charts and insight plots
│   └── 3_CF_SVD_Evaluation.ipynb   ← CF model, SVD, evaluation metrics
│
├── scripts/
│   ├── eda.py                       ← EDA logic as Python script
│   ├── visualization.py             ← Visualisation functions
│   └── cf_svd_evaluation.py         ← CF, SVD, and evaluation pipeline
│
└── README.md
```

---

## Notebook Descriptions

### 1. Exploratory Data Analysis

File:
```text
1_EDA.ipynb
```

Tasks performed:

- Loaded ratings, movies, and users DAT files
- Mapped occupation codes (0–20) to readable occupation names
- Mapped age codes to age group labels (Under 18, 18–24, 25–34, etc.)
- Merged all three files into a single unified dataframe
- Performed data quality checks — missing values, data types, rating range
- Computed key statistics: total ratings, unique users, unique movies, average ratings per user
- Saved `merged_data.csv` for use in subsequent notebooks

---

### 2. Visualisation

File:
```text
2_Visualization.ipynb
```

Charts produced:

| Plot | Insight |
|---|---|
| Rating distribution | How ratings are spread across 1–5 |
| Top 20 most-rated movies | Most popular movies by volume |
| Top 20 highest average-rated movies | Best-loved films (min 50 ratings) |
| Average rating by gender | Whether male/female users rate differently |
| Number of ratings by age group | Most active age group on platform |
| Average rating by age group | Whether age influences rating behaviour |
| Most common genres | Which genres dominate the catalogue |
| Average rating per genre | Which genres score highest |
| User activity distribution | Light vs heavy raters |
| Top 10 most active users | Heaviest raters on platform |
| Average rating by occupation | Occupation-based rating behaviour |

---

### 3. Collaborative Filtering, SVD and Evaluation

File:
```text
3_CF_SVD_Evaluation.ipynb
```

This notebook is divided into four parts:

**Part A — User-Based Collaborative Filtering (original approach)**
- Builds user-movie pivot matrix on full dataset
- Computes pairwise cosine similarity between all users
- Recommends top-10 movies using weighted average of top-5 similar users
- Filters out already-watched movies

**Part B — SVD Matrix Factorisation (upgrade)**
- Splits data: 80% train, 20% test
- Builds user-movie matrix from training data only
- Applies mean-centring to remove per-user rating bias
- Decomposes matrix using `svds` with k=50 latent factors
- Reconstructs full predicted ratings matrix

**Part C — Evaluation Metrics**
- Computes RMSE and MAE on test set predictions
- Plots predicted vs actual ratings scatter chart
- Computes Precision@K and Recall@K across users
- Plots Precision@K and Recall@K curves for K = 5 to 30

**Part D — Comparison**
- Side-by-side recommendation output for the same user from both methods

---

## Feature Engineering

### Occupation Name Mapping

The users dataset contains occupation as a numeric code (0–20). These were mapped to readable labels:

| Code | Occupation |
|---|---|
| 0 | Other |
| 1 | Academic/Educator |
| 4 | College/Grad Student |
| 7 | Executive/Managerial |
| 12 | Programmer |
| 15 | Scientist |
| 17 | Technician/Engineer |
| ... | ... |

### Age Group Mapping

Age codes were converted to human-readable groups for visualisation and demographic analysis:

| Code | Age Group |
|---|---|
| 1 | Under 18 |
| 18 | 18–24 |
| 25 | 25–34 |
| 35 | 35–44 |
| 45 | 45–49 |
| 50 | 50–55 |
| 56 | 56+ |

---

## Handling Data Sparsity

The user-movie matrix has 6,040 users and 3,706 movies — most cells are empty (zero) because each user rates only a small fraction of all movies. This is called the **sparsity problem** and is the main challenge in collaborative filtering.

### How it was addressed:

- **Cosine Similarity (baseline):** Works on raw sparse matrix but degrades in quality when overlap between users is low.
- **SVD Matrix Factorisation:** Decomposes the matrix into latent factors (hidden tastes like "prefers action" or "likes 90s films") that generalise across users and movies even when explicit co-ratings are missing. This is the industry-standard solution to sparsity.
- **Mean-Centring before SVD:** Each user's average rating is subtracted before decomposition to remove the bias of users who consistently rate high or low.

---

## Machine Learning Methods

### User-Based Collaborative Filtering

Memory-based CF approach — no model training required. Finds similar users using cosine similarity and aggregates their ratings.

| Property | Detail |
|---|---|
| Type | Memory-Based Collaborative Filtering |
| Similarity measure | Cosine similarity |
| Number of neighbours | Top 5 (threshold ≥ 0.3) |
| Aggregation | Weighted average by similarity score |
| Cold start handling | None (limitation) |

---

### SVD — Matrix Factorisation

Model-based CF approach — learns latent factors from the training data.

| Property | Detail |
|---|---|
| Type | Model-Based Collaborative Filtering |
| Algorithm | Singular Value Decomposition (truncated) |
| Library | `scipy.sparse.linalg.svds` |
| Latent factors (k) | 50 |
| Mean-centring | Yes — per-user mean subtracted |
| Famous application | Netflix Prize (2009 winning solution) |

---

## Model Evaluation

### Evaluation Metrics

| Metric | Measures | Better when |
|---|---|---|
| RMSE | Root Mean Squared Error — rating prediction accuracy, penalises large errors more | Lower |
| MAE | Mean Absolute Error — average rating prediction error | Lower |
| Precision@K | Of the top-K recommendations, what fraction did the user actually like (rating ≥ 4) | Higher |
| Recall@K | Of all movies the user liked, what fraction appeared in the top-K list | Higher |

### Interpretation Guide

- RMSE < 1.0 is considered good for a 1–5 rating scale
- Precision@10 > 10% is an acceptable baseline for recommender systems
- Recall increases as K increases — the Precision@K vs Recall@K curve shows the tradeoff

---

## How to Run This Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-link>
```

### 2. Navigate to Project Folder

```bash
cd Movie-Recommendation-System
```

### 3. Create a Virtual Environment

```bash
conda create -n movie-rec-env python=3.11
```

### 4. Activate the Environment

```bash
conda activate movie-rec-env
```

### 5. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 6. Download the Dataset

Place the MovieLens 1M dataset files inside the `data/` folder:

```text
data/
├── ratings.DAT
├── movies.DAT
└── users.DAT
```

Dataset available at: [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)

### 7. Run Notebooks in Order

```text
1_EDA.ipynb                 ← Run first (generates merged_data.csv)
→ 2_Visualization.ipynb     ← Run second
→ 3_CF_SVD_Evaluation.ipynb ← Run third
```

> Note: `1_EDA.ipynb` must be run first as it generates `merged_data.csv` which is required by the other two notebooks.

---

## Future Improvements

### Popularity-Based Recommender
A simple but effective baseline that recommends the most-rated or highest-average-rated movies globally. Useful for new users with no rating history (cold start problem). Can be combined with the existing CF model as a fallback.

### Item-Based Collaborative Filtering
Instead of finding similar users, find similar movies. Often more stable than user-based CF since movie tastes change less over time than user behaviour.

### Content-Based Filtering
Use the `genres` column from the movies dataset with TF-IDF vectorisation and cosine similarity to recommend movies similar to what a user has previously liked — without needing other users' data.

### Hybrid Recommender
Combine collaborative filtering (SVD) with content-based filtering using a weighted blend. This is what Netflix and Spotify use in production — it handles cold start for both new users and new movies.

### Hyperparameter Tuning for SVD
Experiment with different values of k (number of latent factors). More factors capture finer-grained tastes but risk overfitting. A proper cross-validation sweep would find the optimal k.

### Deep Learning — Neural Collaborative Filtering (NCF)
Replace SVD with a neural network that learns user and movie embeddings jointly. PyTorch or Keras implementation. Useful when dataset size grows beyond 1M ratings.

### Demographic-Aware Recommendations
The `users.DAT` file contains gender, age, and occupation — features not currently used in the recommendation model. These can personalise recommendations for new users before they have rated enough movies.

### Real-Time API Deployment
Wrap the SVD model in a FastAPI or Flask REST API so the recommendation engine can serve live predictions for a streaming platform frontend.

### Streamlit Dashboard
Build an interactive web application where a user selects their user ID and receives personalised movie recommendations instantly, with visualisations of their rating profile and similar users.

### Cloud Deployment
Deploy the Streamlit or FastAPI application on AWS, GCP, or Heroku for public access and scalability testing.

---

## Key Learnings

This project demonstrates:

- End-to-end data analytics and machine learning workflow on a real-world dataset
- How recommender systems work — from raw ratings to personalised suggestions
- The sparsity problem in collaborative filtering and how SVD solves it
- Difference between Memory-Based CF (cosine similarity) and Model-Based CF (SVD)
- How latent factors represent hidden user preferences without explicit labels
- Proper ML evaluation — why train/test split matters and how to use RMSE, MAE, Precision@K, Recall@K
- Data visualisation for analytics — turning raw data into actionable insights about user behaviour, genre trends, and demographic patterns

---

## Author

**Harshavardhan Gurav**
- 📧 Email: harshavardhangurav839@gmail.com
- 💼 LinkedIn: [harshavardhan-gurav](https://www.linkedin.com/in/harshavardhan-gurav-3284601a2/#:~:text=www.linkedin.com/in/harshavardhan%2Dgurav%2D3284601a2)
