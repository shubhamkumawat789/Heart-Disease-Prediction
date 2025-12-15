# Heart-Disease-Prediction
This project builds an end-to-end Heart Disease Prediction system using a Deep Learning (MLP) model in TensorFlow/Keras, including data preprocessing, model training, evaluation, saving artifacts, and deployment using Streamlit.

## 📊 Dataset Description
The Heart Disease dataset contains clinical and demographic information used to predict the presence of heart disease in patients.

| Column Name | Type        | Description                                                        |
|-------------|-------------|--------------------------------------------------------------------|
| age         | int         | Age of the patient in years                                        |
| sex         | string      | Gender of the patient (Male/Female)                                |
| dataset     | string      | Source dataset (Cleveland, Hungary, Switzerland, VA Long Beach)    |
| cp          | string      | Chest pain type (typical, atypical, non-anginal, asymptomatic)     |
| trestbps    | int         | Resting blood pressure (mm Hg)                                     |
| chol        | int         | Serum cholesterol level (mg/dl)                                    |
| fbs         | bool        | Fasting blood sugar > 120 mg/dl (True/False)                       |
| restecg    | string      | Resting electrocardiographic results                               |
| thalch     | int         | Maximum heart rate achieved                                        |
| exang      | bool        | Exercise-induced angina (True/False)                               |
| oldpeak    | float       | ST depression induced by exercise relative to rest                 |
| slope      | string      | Slope of the peak exercise ST segment                              |
| ca         | int         | Number of major vessels colored by fluoroscopy (0–4)               |
| thal       | string      | Thalassemia condition (normal, fixed defect, reversible defect)    |
| target     | int (binary)| Heart disease status (0 = No Disease, 1 = Disease)                 |

# Process Breakdown

## Step 1: Data Loading
The Heart Disease dataset is loaded and basic exploration is performed to understand features and missing values.

## Step 2: Target Engineering
The target variable is converted into a binary label indicating the presence or absence of heart disease.

## Step 3: Feature Separation
Numerical and categorical features are identified for appropriate preprocessing.

## Step 4: Data Preprocessing
- Numerical features are scaled using StandardScaler
- Categorical features are encoded using OneHotEncoder
- A ColumnTransformer pipeline ensures consistent transformations

## Step 5: Train–Test Split
The dataset is split into training and testing sets using stratified sampling to maintain class balance.

## Step 6: Model Architecture (MLP)
A Deep Learning Multi-Layer Perceptron is built using TensorFlow/Keras with:

- Fully connected (Dense) layers
- ReLU activation
- Dropout for regularization

## Step 7: Model Training
The model is trained using the Adam optimizer and Binary Cross-Entropy loss.

## Step 8: Model Evaluation
Model performance is evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

## Step 9: Model & Pipeline Saving
The trained model and preprocessing pipeline are saved for reuse during inference and deployment.

## Step 10: Deployment
A Streamlit web application is developed to:

- Accept user inputs
- Predict heart disease risk
- Display prediction probability
- Log predictions for future analysis


`

Heart-Disease-Prediction/
│
├── app.py
│   └── Streamlit web application for heart disease prediction
│
├── README.md
│   └── Project documentation
│
├── requirements.txt
│   └── Python dependencies required to run the project
│
├── artifacts/
│   ├── heart_mlp_tf.keras
│   │   └── Trained TensorFlow/Keras MLP model
│   └── preprocessor.joblib
│       └── Saved preprocessing pipeline (scaler + encoder)
│
├── data/
│   └── Heart_Disease.csv
│       └── Original dataset used for training the model
│
├── notebooks/
│   └── Heart_disease.ipynb
│       └── Data analysis, preprocessing, model training, and evaluation
│
├── scripts/
│   └── test_model.py
│       └── Script for testing model inference using saved artifacts
│
├── logs/
│   └── predictions_log.csv
│       └── Logs storing user inputs and prediction results
│
└── .gitignore
    └── Files and folders excluded from version control

`
