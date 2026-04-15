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

Classification Report
───────────────────────────────────────────────────────
                 precision    recall  f1-score   support

Did not survive       0.77      0.92      0.83       110
       Survived       0.81      0.55      0.66        69

       accuracy                           0.78       179
      macro avg       0.79      0.73      0.74       179
   weighted avg       0.78      0.78      0.77       179

### Accessing Seaborne Titanic Dataset: 
import seaborn as sns
df = sns.load_dataset('titanic')

