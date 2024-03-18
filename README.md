# Genome-wide-Methylation-age-prediction

## Overview of the Problem
This project focuses on predicting individual ages based on genome-wide DNA methylation profiles. DNA methylation is a process that can influence gene expression levels and is known to change with age. By analyzing these changes, we aim to develop a machine learning model that can accurately predict the age of individuals from their DNA methylation patterns. This approach has potential applications in forensic analysis, biogerontology, and personalized medicine.

## Description of the Dataset
The dataset used for this project was obtained using the Illumina Infinium 450k Human DNA Methylation Beadchip. This technology allows for the profiling of DNA methylation levels across approximately 450,000 CpG sites in the human genome. Our samples come from human whole blood, covering a wide age range of individuals.

## Input Features: 
DNA methylation beta values at approximately 450,000 CpG sites. These values range from 0 (unmethylated) to 1 (fully methylated).
Outcome (Target Variable): Age of the individual, which is a continuous variable.
Dimensions: The dataset includes methylation profiles for a significant number of individuals (exact number to be specified based on your dataset) and 450,000 features (CpG sites), plus one target variable (age).
How to Run the Code
Dependencies: The analysis requires Python 3.x with the following libraries: pandas, numpy, scikit-learn, xgboost, matplotlib. A requirements.txt file should list all the necessary libraries, and they can be installed using pip install -r requirements.txt.
Running the Analysis: To run the prediction model, use the command python age_prediction.py. This script will load the dataset, preprocess the data, split it into training and testing sets, train the XGBoost model, and finally evaluate its performance.
Decisions Made Along the Way
During the project, several key decisions and trade-offs were made:

## Feature Selection: 
Due to computational constraints, we selected a subset of CpG sites based on their relevance to aging, as determined by preliminary analysis and literature review. This selection aimed to reduce dimensionality and focus on the most informative features.
Model Choice: We opted for XGBoost due to its effectiveness in handling large, sparse datasets and its superior performance in similar prediction tasks. This choice was made considering the trade-off between model complexity and interpretability.
Evaluation Metrics: MSE (Mean Squared Error) was chosen as the primary evaluation metric to emphasize the accuracy of age predictions. However, this focus on MSE might overlook model performance aspects like bias and variance, which are also important.
Example Output

## The output of the project includes:
A trained XGBoost model capable of predicting ages based on DNA methylation profiles.
Performance evaluation of the model, including MSE on the testing set.
A feature importance plot, highlighting the top CpG sites contributing to age prediction.
Citations

## Data Source: Citation for the dataset used, typically including the original study or database that provided the DNA methylation data.
Code and Libraries: Cite the main libraries and any significant codebases or algorithms used in the project. For XGBoost, reference the original paper by Chen and Guestrin (2016).
## Relevant Literature: Include citations for key papers that informed your approach, such as foundational works on DNA methylation and aging.
This project documentation outlines the steps taken to develop a predictive model for age based on DNA methylation data, including the rationale behind key decisions and the outcomes of those choices.
