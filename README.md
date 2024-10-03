# Microsoft Cybersecurity Incidents Detection

## Introduction

In the rapidly evolving cybersecurity landscape, the increasing volume of incidents has overwhelmed Security Operation Centers (SOCs). To address this, there is a pressing need for solutions that can automate or support the remediation process effectively. This project leverages the GUIDE dataset—a groundbreaking collection of real-world cybersecurity incidents—to develop machine learning models for predicting significant cybersecurity incidents and facilitating informed decision-making.

### Problem Statement

As a data scientist at Microsoft, he/she is tasked with enhancing the efficiency of Security Operation Centers (SOCs) by developing a machine learning model that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset, the goal is to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses. The model should be robust enough to support guided response systems in providing SOC analysts with precise, context-rich recommendations, ultimately improving the overall security posture of enterprise environments.

### Business Use Cases

The solution developed in this project can be implemented in various business scenarios, particularly in the field of cybersecurity. Some potential applications include:

**Security Operation Centers (SOCs)**: Automating the triage process by accurately classifying cybersecurity incidents, thereby allowing SOC analysts to prioritize their efforts and respond to critical threats more efficiently.

**Incident Response Automation**: Enabling guided response systems to automatically suggest appropriate actions for different types of incidents, leading to quicker mitigation of potential threats.

**Threat Intelligence**: Enhancing threat detection capabilities by incorporating historical evidence and customer responses into the triage process, which can lead to more accurate identification of true and false positives.

**Enterprise Security Management**: Improving the overall security posture of enterprise environments by reducing the number of false positives and ensuring that true threats are addressed promptly.

## Table of Contents
- [Microsoft Cybersecurity Incidents Detection](#microsoft-cybersecurity-incidents-detection)
  - [Introduction](#introduction)
    - [Problem Statement](#problem-statement)
    - [Business Use Cases](#business-use-cases)
  - [Table of Contents](#table-of-contents)
  - [Domain](#domain)
  - [Dataset Overview](#dataset-overview)
    - [Insight informations](#insight-informations)
    - [Benchmarking](#benchmarking)
    - [Privacy Considerations](#privacy-considerations)
  - [Tools Used](#tools-used)
  - [Approach](#approach)
    - [1. Data Exploration and Understanding](#1-data-exploration-and-understanding)
    - [2. Data Preprocessing](#2-data-preprocessing)
    - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
    - [4. Model Selection and Training](#4-model-selection-and-training)
    - [5. Model Evaluation and Tuning](#5-model-evaluation-and-tuning)
  - [6. Final Evaluation on Test Set](#6-final-evaluation-on-test-set)
  - [7. Model Performance Analysis](#7-model-performance-analysis)
  - [8. Inferences](#8-inferences)
  - [9. Recommendations](#9-recommendations)
  - [Workflow](#workflow)
  - [Contact](#contact)

## Domain

Cybersecurity

## Dataset Overview

GUIDE_train.csv (2.43 GB)
GUIDE_test.csv (1.09 GB)
[Kaggle Link to Dataset](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction)

The GUIDE dataset consists of over 13 million pieces of evidence across three hierarchical levels:

1. **Evidence**: Individual data points supporting an alert (e.g., IP addresses, user details).
2. **Alert**: Aggregated evidences indicating potential security incidents.
3. **Incident**: A comprehensive narrative representing one or more alerts.

### Insight informations

1. **Size**: Over 1 million annotated incidents with triage labels, and 26,000 incidents with remediation action labels.
2. **Telemetry**: Data from over 6,100 organizations, including 441 MITRE ATT&CK techniques.
3. **Training/Testing**: The dataset is divided into a training set (80%) and a test set (20%), ensuring stratified representation of triage grades and identifiers.

### Benchmarking

The GUIDE dataset aims to establish standardized benchmarks for guided response systems:

- **Primary Metric**: Macro-F1 score for incident triage predictions.
- **Secondary Metric**: Precision and recall for remediation action predictions.

### Privacy Considerations

To protect sensitive information, the dataset underwent a stringent anonymization process, including:

- **Pseudo-anonymization**: Sensitive values are hashed using SHA1 to maintain uniqueness without revealing identities.
- **Random ID Replacement**: Hashed values are replaced with randomly generated IDs.
- **Temporal Noise**: Timestamps are modified to prevent re-identification.

## Tools Used

**IDE:** Visual Studio Code

**Programming Language**: Python

**Libraries**: Scikit-learn, Pandas, Matplotlib, Seaborn, NumPy

**Version Control**: Git, Github

## Approach

### 1. Data Exploration and Understanding

**Initial Inspection**:

- Loaded the `GUIDE_train.csv` and 'GUIDE_test.csv' dataset and performed an initial inspection to understand the structure of the data, including the number of features, types of variables (categorical, numerical), and the distribution of the target variable (TP, BP, FP).
- Train Dataset contained - rows x columns (9516837, 45)
- Test Dataset contained - rows x columns (4147992 x 46) 
- one extra column called usage(Public or Private, which won't be utilized as it isn't in our train data)

### 2. Data Preprocessing 

**Handling Missing Data and Duplicates**: Identified missing values and droped columns where missing values were more than 50% of the total rows. Duplicates were removed as well. Conversion of datatypes, for example string time to datetime done as well. 

**Feature Engineering**: Created new features or modified existing ones to improve model performance. Entails the creation and selection of features that significantly contribute to the classification task. This includes transforming raw data into meaningful features, encoding categorical variables, and scaling numerical values to improve model performance.

### 3. Exploratory Data Analysis (EDA)

**EDA**: Used visualizations and statistical summaries to identify patterns, correlations, and potential anomalies in the data. Further a closer look at the distribution of incidents across time were as done. Clearly shows fluctuations across time which would be useful for our model.

![CHEESE!](Images/time1.png)
![CHEESE!](Images/time2.png)
![CHEESE!](Images/time3.png)

And the co-relation heatmap as well to understand co-linearity among the features

![CHEESE!](Images/heatmap.png)

**Multi-colinearity** Removing one of the columns where pairs are highly co-related to avoid multi-colinearity

### 4. Model Selection and Training

**Machine Learning Models**: Experimented with more sophisticated models such as Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM). Each model was tuned using techniques like grid search or random search over hyperparameters.

**Train-Validation Split**: Before diving into model training, split the `train.csv` data into training and validation sets. This allowed for tuning and evaluating the model before final testing on `test.csv`. A typical 70-30 or 80-20 split was used, varying depending on the dataset's size.

**Encoding Categorical Variables**: Converted categorical features into numerical representations using techniques like one-hot encoding or label encoding depending on the nature of the feature and its relationship with the target variable.

**Stratification**: Can use stratified sampling to ensure that both the training and validation sets had similar class distributions, especially since the target variable was imbalanced. But in this case we can attempt to proceed with as is at Random Forest and XGBoost are well equiped with dealing with imbalance data and it also represents real world scenario. We can look into other methods after testing this way and evaluating it's metrics.

### 5. Model Evaluation and Tuning

**Performance Metrics**: Evaluated the model using the validation set, focusing on macro-F1 score, precision, and recall. Analyzed these metrics across different classes (TP, BP, FP) to ensure balanced performance.

**Hyperparameter Tuning**: Fine-tuned hyperparameters based on the initial evaluation to optimize model performance. Adjusted learning rates, regularization parameters, tree depths, and the number of estimators, depending on the model type.

## 6. Final Evaluation on Test Set

**Testing**: Once the model was finalized and optimized, it was evaluated on the `test.csv` dataset. Reported the final macro-F1 score, precision, and recall to assess how well the model generalized to unseen data.

## 7. Model Performance Analysis

**Training Dataset Performance**: 

Trained using ensemble methods XGBoost and Random Forest; Random Forest performed the best.

![CHEESE!](Images/Logistics_train_metrics.png)
![CHEESE!](Images/DecisionTree_train_metrics.png)
![CHEESE!](Images/RF_train_metrics.png)
![CHEESE!](Images/XGB_train_metrics.png)

**Test Dataset Performance**:

Selected the Random Forest Classifier and applied it to the test dataset.

![CHEESE!](Images/RF_test_metrics.png)

## 8. Inferences

**High Training Performance**: The model exhibits very high performance on the training dataset, indicating it has learned the patterns in the data well.

**Good Generalization**: The model performs robustly on the test dataset, suggesting it generalizes well to new, unseen data. The slight decrease in accuracy from training to testing is typical and indicates good generalization without significant overfitting.

**Class-wise Variations**: While the model maintains strong performance across most classes, there is a noticeable drop in performance for benign positive incidents in the test set. This could be an area for further investigation and improvement.
Overall, the Random Forest model demonstrates strong capabilities in classifying cybersecurity incidents, with good generalization to real-world data. Future improvements could focus on enhancing performance for specific classes and continuing to monitor and adjust the model as more data becomes available.

## 9. Recommendations

**Integration into SOC Workflows:**

  -**Enhanced Incident Triage:** Integrate the model into SOC workflows to automate and refine the incident triage process, providing SOC analysts with precise classifications of incidents as TP, BP, or FP.
 
  -**Real-time Analysis:** Deploy the model in real-time environments to assist in immediate incident response, helping analysts prioritize and address security threats more effectively.

**Considerations for Deployment:**

  -**Scalability**: Ensure the model can handle large volumes of data in a production environment, potentially leveraging scalable cloud infrastructure.
 
  -**Real-world Testing:** Conduct extensive testing in a real-world setting to validate model performance and address any operational challenges.
  
  -**Feedback Loop:** Implement a feedback loop to capture analyst insights and adjust the model based on real-world usage and performance metrics.

## Workflow
[Slides](https://docs.google.com/presentation/d/1bzLQj3FN7Tjf427imL5FT2uDhttaregGrXIZJbgHmnE)

## Contact
[LinkedIn](https://www.linkedin.com/in/gengatharan007/)

---
^ [Back to table of contents](#table-of-contents)