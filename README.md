# Genome-wide-Methylation-age-prediction

## Overview: 
This project focuses on predicting individual ages based on genome-wide DNA methylation profiles. DNA methylation is a process that can influence gene expression levels and is known to change with age. By analyzing these changes, we aim to develop a machine learning model that can accurately predict the age of individuals from their DNA methylation patterns. This approach has potential applications in forensic analysis, biogerontology, and personalized medicine.
Description of the Dataset
The dataset used for this project was obtained using the Illumina Infinium 450k Human DNA Methylation Beadchip. This technology allows for the profiling of DNA methylation levels across approximately 450,000 CpG sites in the human genome. The samples come from human whole blood, covering a wide age range of individuals.
## How to run the code (dependencies, etc.)
### Input Features: 
DNA methylation beta values at approximately 450,000 CpG sites. These values range from 0 (unmethylated) to 1 (fully methylated).
### Outcome (Target Variable): 
Age of the individual, which is a continuous variable ranging from 19 to 101.
### Dimensions: 
The dataset includes methylation profiles for a 656 individuals and 450,000 features (CpG sites), plus one target variable (age).
### Dependencies: 
Python 3.11 was used to run the model with the following libraries: biolearn, pandas, numpy, sklearn, xgboost, matplotlib. A requirements.txt file in github should list all the necessary libraries, and they can be installed using pip install -r requirements.txt.
### Running the Analysis: 
To run the prediction model, use the command python xprize.ipynb uploaded in github. This script will load the dataset, preprocess the data, split it into training and testing sets, train the XGBoost model, and finally evaluate its performance using Mean Square Error (MSE).
## Decisions made along the way, including trade-offs 
### Feature Selection: 
All CpG sites were used to trained the model which were split into train and test sets. After training the model, feature importance was plotted to show the CpGs sites that are most relevant to age. 
Model Choice: The code was adopted from a biomarkers-of-aging-challenge analysis sample. The model of elastic net was initially used. But the MSE results showing the performance of the model is 652, which is too high and not prefect. We then opted for XGBoost due to its effectiveness in handling large, sparse datasets and its superior performance in similar prediction tasks. This choice was made considering the trade-off between model complexity and interpretability. After adjusting the parameters of the XGBoost model, including gamma, max_depth, lambda, subsample, colsample_bytree, min_child_weight, learning_rate, and alpha, the lowest MSE we got was 26. This is a much lower score and indicates that XGBoost is a much better model to predict age using the CpGs sites dataset. 
### Evaluation Metrics: 
MSE (Mean Squared Error) was chosen as the primary evaluation metric to emphasize the accuracy of age predictions. However, this focus on MSE might overlook model performance aspects like bias and variance, which are also important. 
## Example output (what does it do?)
Performance evaluation of the model, including MSE on the testing set.
mse: 26.254908175681962
A feature importance plot, highlighting the top CpG sites contributing to age prediction.
  

## Citations 
### Data Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
Or download the data using its package: 
from biolearn.data_library import DataLibrary
data = DataLibrary().get("GSE40279").load()
data.metadata
data.dnam
### Code and Libraries: 
https://bio-learn.github.io/auto_examples/02_challenge_submissions/training_simple_model.html - sphx-glr-auto-examples-02-challenge-submissions-training-simple-model-py

![image](https://github.com/petraliu1006/Genome-wide-Methylation-age-prediction/assets/146908861/74591716-89b7-4f6f-b483-8940d46d344e)
