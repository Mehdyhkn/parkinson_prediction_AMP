# parkinson_prediction_AMP
https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview/evaluation

# Introduction 

The goal of this competition is to predict (The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale) MDS-UPDR scores, which measure progression in patients with Parkinson's disease. 
The MDS-UPDR scores is a comprehensive assessment of both motor and non-motor symptoms associated with Parkinson's. You will develop a model trained on data of protein and peptide levels over time in subjects with Parkinson’s disease versus normal age-matched control subjects.


# Context

Parkinson’s disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions. Unfortunately, there is no current cure—and the disease worsens over time. It's estimated that by 2037, 1.6 million people in the U.S. will have Parkinson’s disease, at an economic cost approaching $80 billion. 
** Research indicates that protein or peptide abnormalities play a key role in the onset and worsening of this disease **. 
Gaining a better understanding of this—with the help of data science—could provide important clues for the development of new pharmacotherapies to slow the progression or cure Parkinson’s disease.

# Evaluation 
- Metric SMAPE + 1 (Symmetric mean absolute percentage error ) 
- <img src="https://latex.codecogs.com/gif.latex?SMAPE= \frac{1}{100} \sum_{t=1}^{n} \frac{|F_t - A-t}{(|A_t| + |F_t|)/2} \text { where At is the actual value and Ft is the forecast value.} " /> 
- For each patient visit where a protein/peptide sample was taken you will need to estimate both their UPDRS scores for that visit and predict their scores for any potential visits 6, 12, and 24 months later. Predictions for any visits that didn't ultimately take place are ignored. 

# Data 

