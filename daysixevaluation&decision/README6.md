# Day 6 — Evaluation, Decision Making, and Condition Formation  
## MLIDOMM: Machine Learning Is the Daughter of Mathematics

---

## Overview

Day 6 focuses on the **decision and evaluation layer of machine learning**, answering a critical question:

**How does a machine decide whether a learned model is good enough to act upon?**

Unlike earlier days that concentrated on parameter learning, this day emphasizes:
- Feature preparation
- Performance measurement
- Decision thresholds
- Condition formation for trust and action

This project is implemented **from scratch using NumPy only**, with a strict separation between **evaluation logic** and **decision logic**.

---

## Mathematical Focus

The work aligns with postgraduate level topics in statistics, optimization, and learning theory.

Key mathematical areas covered:
- Feature scaling and representation
- Loss functions and risk estimation
- Bias variance tradeoff
- Decision thresholds
- Generalization and conditioning

---

## Project Structure
Day6_MLIDOMM/
│
├── evaluation.py
│ └── Model training, prediction, and performance evaluation
│
├── decision.py
│ └── Threshold based decision analysis
│
└── README.md

## Design

The project follows a **two layer architecture**:

### 1. Evaluation Layer
Responsible for statistical assessment only.
- Trains the model
- Generates predictions
- Computes error metrics
- Returns raw outputs

### 2. Decision Layer
Responsible for action and judgment.
- Consumes predictions
- Applies decision thresholds
- Computes true positives, false positives, false negatives, and true negatives

This separation mirrors **real world machine learning systems** and enforces clarity between inference and action.

---

## Implemented Concepts

### Feature Representation
- Synthetic data generation
- Numerical stability awareness

---

### Performance Metrics

**Mean Squared Error**


**Mean Absolute Error**

---




---

## How to Run the Project

### Requirements
- Python 3.8 or higher
- NumPy

