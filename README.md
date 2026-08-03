# Heart Disease Classification using K-Nearest Neighbors (KNN)

This repository contains a Jupyter Notebook demonstrating a comprehensive machine learning workflow for classifying heart disease using the K-Nearest Neighbors (KNN) algorithm. The project emphasizes the importance of appropriate evaluation metrics (specifically Recall), hyperparameter tuning, and preventing data leakage using Scikit-Learn Pipelines.

## Overview

The primary goal of this project is to build a predictive model while prioritizing the **Recall** metric. In medical diagnoses like heart disease prediction, minimizing false negatives (predicting a patient is healthy when they actually have the disease) is strictly more critical than overall accuracy. 

## Dependencies

Ensure you have the following Python libraries installed to run the notebook:
* pandas
* scikit-learn

## Dataset

The project relies on a dataset named `1-heart.csv`. The data is separated into:
* **Features (X):** All columns excluding the target.
* **Target (y):** The column named `target`, which contains the binary classification labels.

## Project Workflow

### 1. Data Preprocessing
* **Train-Test Split:** The dataset is divided into 80% training data and 20% testing data (`random_state=42`).
* **Feature Scaling:** `StandardScaler` is applied to normalize feature ranges. This is a critical step for distance-based algorithms like KNN to ensure no single feature disproportionately influences the distance calculations.

### 2. Manual Model Evaluation
The KNN classifier is initially instantiated and tested manually across multiple values of *K* (3, 5, 7, 9, and 11) to observe the trade-offs between Accuracy, Precision, and Recall.

### 3. Automated Hyperparameter Tuning
`GridSearchCV` is utilized with 5-fold cross-validation to programmatically find the optimal *K* value (`n_neighbors`). 
* The notebook demonstrates that optimizing for default Accuracy selects *K=5*.
* Changing the scoring parameter to `scoring="recall"` correctly adjusts the optimal parameter to *K=7*.

### 4. Pipeline Implementation (Best Practice)
A Scikit-Learn `Pipeline` is constructed to chain the `StandardScaler` and `KNeighborsClassifier` together. This represents industry best practice because it ensures scaling is applied independently within each cross-validation fold, completely eliminating the risk of data leakage.

## Results

The final pipeline model, optimized specifically for Recall, yielded the following performance metrics on the unseen test set:-

| Metric | Score |
| :--- | :--- |
| **Best Parameter** | K = 7 |
| **Recall** | ~0.906 |
| **Accuracy** | ~0.918 |
| **Precision** | ~0.935 |

## Usage

To run this project, simply place the `1-heart.csv` file in the same directory as the Jupyter Notebook, install the required dependencies, and execute the cells sequentially.
