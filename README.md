# ML Lifecycle & Deployment – Lab 3 and Final

This repository contains my end-to-end machine learning lifecycle project for Lab 3, extended through deployment using MLflow, FastAPI, Docker, and GitHub.

The project demonstrates how evaluation choices in Lab 3 led to a defensible final model, and how that model is tracked, versioned, and served in a reproducible way.

Project Overview

Goal:
Build, evaluate, track, and deploy a machine learning model using best-practice ML Ops tools.

Key components:

Experimentation and model selection (Lab 3)

MLflow experiment tracking and model logging

Reproducible training via MLproject

FastAPI model serving

Optional Docker containerization

Version control with GitHub

Final Model (from Lab 3)

Dataset: Breast Cancer Wisconsin

Task: Binary classification (malignant vs benign)

Model: Support Vector Classifier (RBF kernel)

Why SVC:

Performed consistently well under stratified and nested cross-validation

Sensitive to hyperparameters (C, gamma), making it ideal to demonstrate proper tuning

Nested CV provided the most honest performance estimate

Primary Metric: ROC-AUC

The deployed model is the final SVC pipeline selected in Lab 3 after evaluating multiple cross-validation strategies.
