# Loan_Analysis-Predictive-Analysis-
The Loan Approval Prediction Project uses a Random Forest Classifier to predict loan approval based on applicant data like income, loan amount, and credit history. The project includes data preprocessing, feature encoding, and model evaluation to build an accurate model that automates and improves the loan approval process.






#### Overview

The Loan Approval Prediction project is a machine learning-based predictive analysis task. The primary goal is to build a model that can predict whether a loan will be approved or not based on the applicant's information.
This project uses a Random Forest Classifier to train the model on historical loan application data and then evaluates the model's performance.

 Project Structure


LoanApprovalPrediction/
 dataset : -  LoanApproval.csv
   
 notebooks/
 data_preprocessing.ipynb # Data preprocessing steps model_training.ipynb     # Model training and evaluatio


### Dataset
The dataset contains information about loan applicants, including various demographic and financial features. Each entry in the dataset includes:

ApplicantIncome: Income of the applicant.
CoapplicantIncome: Income of the co-applicant.
LoanAmount: The loan amount requested.
Loan_Amount_Term: Term of the loan in months.
Credit_History: Credit history of the applicant (1 means all debts are paid, 0 means otherwise).
Gender: Gender of the applicant.
Married: Marital status of the applicant.
Education: Educational background of the applicant.
Self_Employed: Whether the applicant is self-employed.
Property_Area: Area type of the property (Urban, Semiurban, Rural).
Loan_Status: Whether the loan was approved (Y) or not (N)

### Prerequisites:-
 Before you begin, ensure you have the following software installed:

- Python 3.7+
- pip (Python package installer)

Install Dependencies
  To install the required Python packages, navigate to the project directory and run:

pip install -r requirements.txt


### Data Preprocessing
   Data preprocessing is crucial for preparing the data before feeding it into the machine learning model. The preprocessing steps include:
### Handling Missing Values:
  Missing values in the Gender column are filled with the most frequent value (mode).
  Missing values in the LoanAmount column are filled with the mean value.
### Feature Engineering:
  Creating a TotalIncome feature by combining ApplicantIncome and CoapplicantIncome.
  Applying logarithmic transformation to LoanAmount and TotalIncome to reduce skewness.
### Encoding Categorical Variables:
  Categorical variables like Gender, Married, and Education are label-encoded to convert them into numerical values.
### Feature Scaling:
  The features are standardized using StandardScaler to have a mean of 0 and a standard deviation of 1.
### Model Training:
  The model training script (train_model.py) uses a RandomForestClassifier to train the model. Key steps include:
### Splitting the Data:
  The dataset is split into training and test sets using train_test_split.
### Training the Model:
  The RandomForestClassifier is trained on the preprocessed training data.
### Evaluating the Model:
  The model's accuracy is evaluated using the test set. The accuracy_score function from the metrics module is used to calculate the accuracy.


How to Run the Project
Preprocess the Data:

Run the data preprocessing notebook or script to clean and prepare the data for modeling.
Train the Model:

Run the model training notebook or script to train the RandomForestClassifier on the training data.
Evaluate the Model:

After training, evaluate the model on the test set and print out the accuracy.

### Example Commands

```python
# Train the model
python src/train_model.py

# Evaluate the model
python src/evaluate_model.py
```

## Results

After training, the model is evaluated on the test data, and the accuracy is printed out. You can further analyze the predictions and feature importance to understand the model's decision-making process.

## Contributing

If you have suggestions or want to contribute to this project, feel free to open a pull request or issue on GitHub.



---

This README provides an overview of the Loan Approval Prediction project, detailing its structure, setup, and how to use the provided code to train and evaluate the model.
