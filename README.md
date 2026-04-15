# Titanic-Decision-Tree-Prediction
This project showcases a foundational machine learning (ML) project that predicts whether passengers on the Titanic would survive or not. It leverages the seaborn built-in Titanic dataset and Claude AI to create a decision tree model. 

This notebook contains a full pipeline of the ML model:
- Exploratory Data Analysis
- Preprocessing
- Model Training
- Evaluation
- Decision Tree Visualization
- Manual Prediction (for specific passenger characteristics)

### Key Findings
Train accuracy : 0.836 | Test  accuracy : 0.777
Tree depth     : 4 | Leaf nodes     : 14

The model predicted 77.7% of passengers outcome correctly (~6% higher than on train data)

Precision
Did Not Survive:   0.77
Survived:          0.81

Recall
Did Not Survive:   0.92
Survived:          0.55



### Accessing Seaborne Titanic Dataset: 
import seaborn as sns
df = sns.load_dataset('titanic')

