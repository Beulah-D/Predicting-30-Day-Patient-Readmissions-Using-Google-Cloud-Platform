# Predicting-30-Day-Patient-Readmissions-Using-Google-Cloud-Platform
Building end-to-end training and inference pipelines to predict hospital readmissions using healthcare data

## Overview 
 ### 🚀Project Goal:

- Automating Machine Learning workflows for Predicting 30-Day Hospital Readmissions
- This project leverages Machine Learning Operations (MLOps) to automate the entire lifecycle of predictive models—from data preparation and model training to deployment and monitoring. The goal is to predict the likelihood of hospital readmissions within 30 days using a healthcare dataset containing patient demographic, medical, and treatment-related information.
- By integrating MLOps practices, this project ensures:
  - Efficient Model Training & Deployment: Streamlined workflows for continuous integration and deployment (CI/CD).
  - Scalability & Monitoring: Scalable solutions with real-time performance monitoring and maintenance.
  - Improved Healthcare Outcomes: Supporting healthcare providers in identifying high-risk patients and reducing readmission rates.
 ### 💡 Why This Project Matters: 

- Early Risk Detection: Identifies patients at high risk of readmission for timely intervention.
- Better Patient Outcomes: Supports targeted care strategies to improve treatment and recovery.
- Cost Reduction: Helps hospitals avoid costly readmissions and financial penalties under the Affordable Care Act.
### 📊 Dataset Overview

- Source: Provided as part of academic coursework at the University of Southern California.
- Size: 8,038 rows × 19 columns
- Target Variable: Readmission within 30 Days → Binary (1 = Yes, 0 = No)
- Key Variables:
   - Demographic: PatientID, Age, Gender (Male/Female), Ethnicity (Caucasian, Hispanic, African American, Other), Hospital ID (Hosp1, Hosp2, Hosp3)
   - Clinical: Height (m), Weight (kg), Adjusted Weight (kg), BMI, Smoker (Yes/No), Has Diabetes (Yes/No), Has Hypertension (Yes/No)
   - Behavioral & Lifestyle: Exercise Frequency (None, Occasional, Regular), Diet Type (Balanced, High-fat, Vegetarian, Other)
   - Medical History & Treatment: Number of Prior Visits, Medications Prescribed, Length of Stay, Type of Treatment (None, Minor Surgery, Major Surgery, Other)
### ⚙️ Feature Engineering & Data Preparation
#### 1. Data Cleaning
- Missing Values:
  - Number of Prior Visits: Imputed with the median due to positive skew (0.57).
  - Medications Prescribed: Imputed with the mean due to symmetric distribution.
- Outlier Handling:
  - Age: Winsorized at the 99th percentile to reduce the impact of extreme values.
  - Adjusted Weight & Number of Prior Visits: Capped at the 95th percentile to prevent distortion.
- Data Transformation:
  - Length of Stay: Applied log transformation (log(1 + x)) to correct high positive skew (1.90).
#### 2. Feature Encoding:
  - Binary Encoding:
     - Gender: Mapped Male → 0, Female → 1.
     -  Smoker: Converted to integers (0 = No, 1 = Yes).
  - One-Hot Encoding: Applied to multi-class categorical variables: Ethnicity, Diet Type, Type of Treatment, Exercise Frequency.
#### 3. Feature Selection
- Dropped Features:
  - Hospital ID: Not correlated with the target variable.
  - Weight (Kg): Highly correlated with Adjusted Weight (0.88) → avoided multicollinearity.
  - Patient ID: Removed as it’s a unique identifier with no predictive value.
#### 4. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the target classes.
#### 5. Final Dataset Summary
- After preprocessing, the dataset expanded to 10,630 rows × 26 columns.

### Training Pipeline:
<img width="458" alt="image" src="https://github.com/user-attachments/assets/ae97111d-6bc3-47f3-ba9f-24340824edf0" />

- This pipeline automates the end-to-end process of predicting hospital readmissions within 30 days. It begins with       data preparation, handling missing values, outliers, and dropping irrelevant columns. The preprocessing step includes   encoding categorical variables, applying SMOTE for class imbalance, and scaling numerical features. The data is then    split into training and validation sets. The model training component builds a predictive model, and the evaluation     component assesses its performance using metrics like accuracy, F1 score, and AU-PRC. Artifacts (processed data,        scaler, trained model, and metrics) are stored for reproducibility and deployment on Google Cloud. The model achieved an F1 score of 0.78 on the training dataset, indicating a strong balance between precision and recall in predicting 30-day hospital readmissions
<img width="287" alt="Screenshot 2025-01-14 at 6 16 40 PM" src="https://github.com/user-attachments/assets/a96169d1-2a94-4eb8-940c-57f20299926f" />

### Inference Pipeline:
<img width="612" alt="image" src="https://github.com/user-attachments/assets/e57f794b-0e1b-4ac8-97c7-468aee7fdb6e" />

- The Healthcare Readmissions Inference Pipeline automates predicting 30-day hospital readmissions using a trained model. It begins with data preparation, cleaning and handling missing values in the test dataset. Next, the data is normalized using the scaler from the training pipeline to ensure consistent feature scaling. Finally, the trained model is used to generate predictions on the processed data, outputting patient IDs with their readmission predictions. All artifacts, including the scaled data and predictions, are stored in Google Cloud for easy access and further analysis.
  
## Conclusion: 
- Feature engineering decisions that improved the model: 
  - Log Transformed and imputed missing values while handling outliers, particularly those with positive skewness. Winsorization effectively managed extreme outliers in variables like Age.
  - Removed unnecessary columns like  and features with high correlation to other predictors, reducing multicollinearity like Weight
  - Applying One-Hot Encoding and Binary Encoding to categorical variables contributed to improving the model's performance
- XGBoost Classifier is a better model when the right hyperparameters are used
- Adopted XGBoost Classifier(Ensemble model) as it performed better on the test dataset with an F1 score of 0.78 compared to other models, such as Logistic Regression, which achieved an F1-score of 0.64.

