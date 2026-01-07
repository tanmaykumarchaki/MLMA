<<<<<<< HEAD
# Day 3 — Regularization in Linear Regression  
### MLIDOMM: *Machine Learning Is the Daughter of Mathematics*

---

## 📌 Objective

Day 3 focuses on **Regularization**, a fundamental concept that connects **linear algebra, calculus, optimization, and statistical learning theory**.

This mini project demonstrates, both **mathematically and empirically**:
- Why Ordinary Least Squares (OLS) tends to overfit
- How **Ridge (L2)** and **Lasso (L1)** regularization improve generalization
- The effect of regularization on numerical stability and model complexity

---

## 🧠 Mathematical Foundation

### 1. Ordinary Least Squares (OLS)

The Mean Squared Error loss is defined as:


**Limitation**
- When features are highly correlated, \( X^T X \) becomes ill-conditioned
- Leads to unstable coefficients and poor generalization

---

### 2. Ridge Regression (L2 Regularization)

Adds a penalty on the squared magnitude of coefficients

**Effect**
- Improves numerical conditioning
- Reduces variance
- Introduces small bias for better generalization

---

### 3. Lasso Regression (L1 Regularization)


**Key Properties**
- No closed-form solution
- Produces sparse coefficients
- Performs implicit feature selection

---

## 🛠 Mini Project Description

A synthetic dataset with **high multicollinearity** is created to expose the weaknesses of OLS and justify regularization.

### Implementations:
- Ordinary Least Squares (manual)
- Ridge Regression (manual closed-form)
- Lasso Regression (solver-based)
- Train/Test evaluation
- Coefficient comparison visualization

---

## 📂 Project Structure
Day-3-Regularization/
│
├── regularization_demo.py
├── README.md


---

## 📊 Results Summary

| Model | Training Error | Test Error | Coefficient Stability |
|------|---------------|------------|-----------------------|
| OLS  | Very Low | High | Unstable |
| Ridge | Moderate | Lower | Stable |
| Lasso | Moderate | Competitive | Sparse |

---

## 📈 Visualization

The project visualizes:
- Coefficient shrinkage across OLS, Ridge, and Lasso
- The stabilizing effect of regularization on learned parameters

---

## 🎓 PG Entrance Mapping

- Differential Calculus → Optimization
- Linear Algebra → Conditioning and eigenvalues
- Convexity → Guaranteed global minima
- Bias–Variance Tradeoff → Statistical learning theory

---

## 🚀 Key Takeaway

> Regularization is not a heuristic — it is a mathematical necessity  
> for stable and generalizable machine learning models.

---

## 🔖 Tags

Machine Learning · Mathematics · Regularization · Ridge Regression · Lasso · Optimization · From Scratch · MLIDOMM
=======
# Day 3 — Regularization in Linear Regression  
### MLIDOMM: *Machine Learning Is the Daughter of Mathematics*

---

## 📌 Objective

Day 3 focuses on **Regularization**, a fundamental concept that connects **linear algebra, calculus, optimization, and statistical learning theory**.

This mini project demonstrates, both **mathematically and empirically**:
- Why Ordinary Least Squares (OLS) tends to overfit
- How **Ridge (L2)** and **Lasso (L1)** regularization improve generalization
- The effect of regularization on numerical stability and model complexity

---

## 🧠 Mathematical Foundation

### 1. Ordinary Least Squares (OLS)

The Mean Squared Error loss is defined as:


**Limitation**
- When features are highly correlated, \( X^T X \) becomes ill-conditioned
- Leads to unstable coefficients and poor generalization

---

### 2. Ridge Regression (L2 Regularization)

Adds a penalty on the squared magnitude of coefficients

**Effect**
- Improves numerical conditioning
- Reduces variance
- Introduces small bias for better generalization

---

### 3. Lasso Regression (L1 Regularization)


**Key Properties**
- No closed-form solution
- Produces sparse coefficients
- Performs implicit feature selection

---

## 🛠 Mini Project Description

A synthetic dataset with **high multicollinearity** is created to expose the weaknesses of OLS and justify regularization.

### Implementations:
- Ordinary Least Squares (manual)
- Ridge Regression (manual closed-form)
- Lasso Regression (solver-based)
- Train/Test evaluation
- Coefficient comparison visualization

---

## 📂 Project Structure
Day-3-Regularization/
│
├── regularization_demo.py
├── README.md


---

## 📊 Results Summary

| Model | Training Error | Test Error | Coefficient Stability |
|------|---------------|------------|-----------------------|
| OLS  | Very Low | High | Unstable |
| Ridge | Moderate | Lower | Stable |
| Lasso | Moderate | Competitive | Sparse |

---

## 📈 Visualization

The project visualizes:
- Coefficient shrinkage across OLS, Ridge, and Lasso
- The stabilizing effect of regularization on learned parameters


---

## 🚀 Key Takeaway

> Regularization is not a heuristic — it is a mathematical necessity  
> for stable and generalizable machine learning models.

---

## 🔖 Tags

Machine Learning · Mathematics · Regularization · Ridge Regression · Lasso · Optimization · From Scratch · MLIDOMM

>>>>>>> 565ac4466e9c3e6d97488af59fe8d8f8b2fe907e
