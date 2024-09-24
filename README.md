Heart Disease Classification

Project Overview

This project aims to build a machine learning model that can classify patients as having heart disease or not based on medical attributes. The dataset used contains several features related to a patient's health, such as age, cholesterol levels, and maximum heart rate. By analyzing these features, the model predicts the likelihood of heart disease, providing a tool that could assist in early diagnosis and prevention.

Dataset

The dataset used for this project is the **UCI Heart Disease Dataset**, which contains data on 303 patients with a mix of 13 clinical features:

- **Age**: Age of the patient (years)
- **Sex**: Gender (1 = male, 0 = female)
- **Chest Pain Type**: (0 to 3, where 0 = typical angina, 1 = atypical angina, etc.)
- **Resting Blood Pressure**: Measured in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: (1 if > 120 mg/dl, 0 otherwise)
- **Resting ECG**: Results from resting electrocardiography (0 to 2)
- **Maximum Heart Rate Achieved**: Maximum heart rate
- **Exercise-Induced Angina**: (1 = yes, 0 = no)
- **ST Depression**: Depression induced by exercise relative to rest
- **Slope of ST Segment**: Slope of the peak exercise ST segment (0 to 2)
- **Number of Major Vessels**: (ranging from 0 to 4) colored by fluoroscopy
- **Thalassemia**: (0 = normal, 1 = fixed defect, 2 = reversible defect)

The target variable is **Heart Disease** (1 = disease, 0 = no disease).

## Installation and Dependencies

To run this project, clone the repository and ensure you have the following dependencies installed:

```bash
git clone https://github.com/your-username/heart-disease-classification.git
cd heart-disease-classification
```

You can install the necessary packages by running:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- **Python 3.7+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook** (for experimentation)

## Model Training

The following machine learning algorithms were explored:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting Classifier**

Each model was evaluated using **accuracy**, **precision**, **recall**, and the **F1 score**. Cross-validation and hyperparameter tuning were employed to improve performance.

## Usage

To train the model, use the following command:

```bash
python train_model.py
```

This script will load the dataset, preprocess the data, train the model, and save the trained model in the `models/` directory.

To make predictions on new data:

```bash
python predict.py --input data/new_patient_data.csv
```

## Evaluation

The best-performing model, **Random Forest Classifier**, achieved the following metrics on the test set:
- **Accuracy**: 85.3%
- **Precision**: 82.5%
- **Recall**: 84.0%
- **F1 Score**: 83.2%

The **confusion matrix** shows a balanced ability to predict both positive (patients with heart disease) and negative (patients without heart disease) cases.

## Results and Insights

The model shows that certain features, such as **maximum heart rate**, **chest pain type**, and **exercise-induced angina**, are key indicators of heart disease. The model can be further fine-tuned or deployed as part of a broader healthcare system for early detection.

## Future Improvements

- **Model Deployment**: Develop a web app using Flask or Streamlit for easier interaction.
- **Deep Learning**: Experiment with deep neural networks for potentially better performance.
- **Feature Engineering**: Further explore feature transformations and interactions to improve model accuracy.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. Make sure to write tests for new features and maintain code quality.

