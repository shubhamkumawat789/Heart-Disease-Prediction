# Heart-Disease-Prediction
This project builds an end-to-end Heart Disease Prediction system using a Deep Learning (MLP) model in TensorFlow/Keras, including data preprocessing, model training, evaluation, saving artifacts, and deployment using Streamlit.

## Process Breakdown

### Step 1: Data Loading
The Heart Disease dataset is loaded and basic exploration is performed to understand features and missing values.

### Step 2: Target Engineering
The target variable is converted into a binary label indicating the presence or absence of heart disease.

### Step 3: Feature Separation
Numerical and categorical features are identified for appropriate preprocessing.

### Step 4: Data Preprocessing
- Numerical features are scaled using StandardScaler
- Categorical features are encoded using OneHotEncoder
- A ColumnTransformer pipeline ensures consistent transformations

### Step 5: Train–Test Split
The dataset is split into training and testing sets using stratified sampling to maintain class balance.

### Step 6: Model Architecture (MLP)
A Deep Learning Multi-Layer Perceptron is built using TensorFlow/Keras with:

- Fully connected (Dense) layers
- ReLU activation
- Dropout for regularization

### Step 7: Model Training
The model is trained using the Adam optimizer and Binary Cross-Entropy loss.

### Step 8: Model Evaluation
Model performance is evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

### Step 9: Model & Pipeline Saving
The trained model and preprocessing pipeline are saved for reuse during inference and deployment.

### Step 10: Deployment
A Streamlit web application is developed to:

- Accept user inputs
- Predict heart disease risk
- Display prediction probability
- Log predictions for future analysis
