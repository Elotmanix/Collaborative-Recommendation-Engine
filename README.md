

# Collaborative Recommendation Engine with a Restricted Boltzmann Machine (RBM)

**Module**: Introduction to Machine Learning  
**Academic Institution**: École Centrale Casablanca  
**Academic Year**: 2024/2025  
**Authors**: Adam LOZI, Hamza EL OTMANI, Reda BENKIRANE, Aymane EL FAHSI, Mohamed EL IDRISSI  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Challenges and Limitations](#challenges-and-limitations)
8. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a **movie recommendation system** using a **Gaussian-Bernoulli Restricted Boltzmann Machine (RBM)** for collaborative filtering. The model predicts user ratings for movies they haven't rated, generating personalized recommendations based on their historical preferences and those of similar users.

### Objectives
- Fill in missing ratings within a user-movie ratings matrix by predicting probable ratings.
- Recommend the top 10 movies for each user based on predicted ratings.

---

## Dataset

The **MovieLens** dataset is used in this project, which contains user ratings for movies. The data is structured as a user-item matrix with the following dimensions:
- **Users**: 610
- **Movies**: 9,742
- **Ratings**: Normalized to the \[0, 1\] range.

---

## Model Architecture

The model is a **Gaussian-Bernoulli RBM**:
- **Visible Units**: Represent each movie, taking continuous normalized ratings as input.
- **Hidden Units**: Capture latent features influencing user preferences.
  
The RBM learns to predict missing entries in the ratings matrix by reconstructing user preferences. The project employs the **Contrastive Divergence (CD-k)** algorithm for training, with k=1.

---

## Installation and Setup

### Prerequisites
- Python 3.7+
- Required packages: `numpy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Elotmanix/Collaborative-Recommendation-Engine.git
   cd Collaborative-Recommendation-Engine
   ```


2. **Download the MovieLens dataset**:
   - Download the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and place it in the `data/` directory.






## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Hit Rate@K**: Measures if a recommended item is in the top-K items the user actually rated.
- **Precision@K**: The proportion of recommended items that are relevant.
- **Recall@K**: The proportion of relevant items that are recommended.

Example results:
- **Hit Rate@10**: 0.9918
- **Precision@10**: 0.6016
- **Recall@10**: 0.0868

---

## Challenges and Limitations

1. **Choosing the Appropriate RBM Variant**: We chose a Gaussian-Bernoulli RBM to handle continuous ratings, but this choice introduced complexity in parameter tuning.
2. **Hyperparameter Tuning**: Optimal performance required extensive experimentation with hyperparameters like the learning rate, number of hidden units, and variance.
3. **Balancing Complexity and Efficiency**: A high number of hidden units can improve accuracy but at a computational cost. We had to balance these factors to ensure feasible training times.
4. **Interpretability**: Analyzing latent features learned by the RBM can be challenging, adding complexity to evaluating model predictions.

---

## Future Improvements

- **Hybrid Model Approaches**: Integrate RBMs with other collaborative filtering methods or neural architectures to improve scalability and accuracy.
- **Enhanced Hyperparameter Optimization**: Use more advanced techniques like Bayesian optimization to fine-tune hyperparameters.
- **Additional Evaluation Metrics**: Expand evaluation with metrics like RMSE for more comprehensive performance assessment.




## Acknowledgments

- MovieLens dataset by GroupLens Research.
- École Centrale Casablanca for providing academic guidance.

