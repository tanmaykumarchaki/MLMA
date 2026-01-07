# Day 4 — Maximum Likelihood Estimation for Linear Regression  
### MLIDOMM: Machine Learning Is the Daughter of Mathematics

---

## 📌 Objective

Day 4 establishes the **probabilistic foundation of Linear Regression** by showing that the commonly used Mean Squared Error loss arises naturally from **Maximum Likelihood Estimation (MLE)** under Gaussian noise assumptions.

This project demonstrates, through mathematics and code, that:

> Minimizing Mean Squared Error is equivalent to maximizing the likelihood of the data.

---

## 🛠 Mini Project Description

This project implements:

- Synthetic data generation with Gaussian noise
- Closed-form Maximum Likelihood Estimation
- Mean Squared Error computation
- Log likelihood computation
- Numerical verification of MSE–MLE equivalence
- Visualization of loss and likelihood curves

All implementations are done **from scratch using NumPy**, without ML libraries.

---

## 📂 Project Structure

Day-4-MLE-Linear-Regression/
│
├── mle_linear_regression.py
├── README.md

yaml
Copy code

---

## 📊 Key Observations

- MSE reaches its minimum at the same parameter value where log likelihood reaches its maximum
- Linear Regression parameters are the most probable parameters under Gaussian noise
- The optimization view and probabilistic view are mathematically identical

---

## 🎓 Academic and PG Entrance Mapping

- Probability density functions
- Gaussian distribution
- Log likelihood
- Optimization via differentiation
- Linear algebra and matrix calculus

---

## 🚀 Key Takeaway

> Linear Regression minimizes Mean Squared Error because it is the Maximum Likelihood Estimator under Gaussian noise assumptions.

---

## 🔖 Tags

Machine Learning · Probability · Statistics · Maximum Likelihood Estimation · Linear Regression ·
