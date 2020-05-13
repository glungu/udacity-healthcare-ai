# Project Instructions

1. Project Instructions & Prerequisites
1. Learning Objectives
1. Steps to Completion

## 1. Project Instructions

Context: You are a data scientist for an exciting unicorn healthcare startup that has created 
a groundbreaking diabetes drug that is ready for clinical trial testing. 
It is a very unique and sensitive drug that requires administering the drug over 
at least 5-7 days of time in the hospital with frequent monitoring/testing 
and patient medication adherence training with a mobile application. 
You have been provided a patient dataset from a client partner and are tasked with 
building a predictive model that can identify which type of patients the company 
should focus their efforts testing this drug on. 
Target patients are people that are likely to be in the hospital for this duration 
of time and will not incur significant additional costs for administering this drug 
to the patient and monitoring.

In order to achieve your goal you must build a regression model that can predict 
the estimated hospitalization time for a patient and use this to select/filter patients 
for your study.

Expected Hospitalization Time Regression Model: Utilizing a synthetic dataset
(denormalized at the line level augmentation) built off of the UCI Diabetes readmission dataset, 
students will build a regression model that predicts the expected days of hospitalization 
time and then convert this to a binary prediction of whether to include or exclude 
that patient from the clinical trial.

This project will demonstrate the importance of building the right data representation 
at the encounter level, with appropriate filtering and preprocessing/feature engineering 
of key medical code sets. This project will also require students to analyze and 
interpret their model for biases across key demographic groups.

### Dataset

Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available 
datasets and some datasets require training and approval. 
So, for the purpose of this exercise, we are using a dataset from UC Irvine 
that has been modified for this course. 
Please note that it is limited in its representation of some key features 
such as diagnosis codes which are usually an unordered list in 835s/837s 
(the HL7 standard interchange formats used for claims and remits).

[https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) 
Data Schema The dataset reference information can be 
[https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references](https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references). 
There are two CSVs that provide more details on the fields and some of the mapped values.

### Project Submission

When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "student_project_submission.ipynb" and save another copy as an HTML file by clicking "File" -> "Download as.."->"html". Include the "utils.py" and "student_utils.py" files in your submission. The student_utils.py should be where you put most of your code that you write and the summary and text explanations should be written inline in the notebook. Once you download these files, compress them into one zip file for submission in the Udacity Classroom.

### Prerequisites

* Intermediate level knowledge of Python
* Basic knowledge of probability and statistics
* Basic knowledge of machine learning concepts
* Installation of Tensorflow 2.0 and other dependencies
(conda environment.yml or virtualenv requirements.txt file provided)

### Environment Setup

For step by step instructions on creating your environment, please go to 
[https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/README.md](https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/README.md)

## 2. Learning Objectives 

By the end of the project, you will be able to:

1. Use the Tensorflow Dataset API to scalably extract, transform, and load datasets 
and build datasets aggregated at the line, encounter, and patient data levels(longitudinal)
1. Analyze EHR datasets to check for common issues 
(data leakage, statistical properties, missing values, high cardinality) 
by performing exploratory data analysis.
1. Create categorical features from Key Industry Code Sets (ICD, CPT, NDC) 
and reduce dimensionality for high cardinality features by using embeddings
1. Create derived features(bucketing, cross-features, embeddings) utilizing Tensorflow 
feature columns on both continuous and categorical input features
1. Use the Tensorflow Probability library to train a model that provides uncertainty 
range predictions that allow for risk adjustment/prioritization and triaging of predictions
1. Analyze and determine biases for a model for key demographic groups by evaluating 
performance metrics across groups by using the Aequitas framework

## 3. Steps to Completion

Please follow all of the direction in the Jupyter Notebook file in classroom workspace 
or from the Github Repo if you decide to use your own environment to complete the project.

You complete the following steps there:

1. Data Analysis
1. Create Categorical Features with TF Feature Columns
1. Create Continuous/Numerical Features with TF Feature Columns
1. Build Deep Learning Regression Model with Sequential API and TF Probability Layers
1. Evaluating Potential Model Biases with Aequitas Toolkit

## Project Submission

When submitting this project, make sure to run all the cells before saving the notebook. 
Save the notebook file as "student_project_submission.ipynb" and save another copy as an 
HTML file by clicking "File" -> "Download as.."->"html". 
Include the "utils.py" and "student_utils.py" files in your submission. 
The student_utils.py should be where you put most of your code that you write and the 
summary and text explanations should be written inline in the notebook. 
Once you download these files, compress them into one zip file for submission in the 
Udacity Classroom.

Once you have completed your project please

1. Make sure the project meets all of the specifications on the Project Rubric
1. If you are working in directly in our workspaces, you can submit your project directly there
1. If you are working in your own environment or if you have issues submitting directly 
in the workspace, please zip up your flies and submit them that way.

Best of Luck on the project. Remember that you can use the resources provided 
in the student hub or talk with you mentor if you have questions too.

