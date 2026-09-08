# 🎓 Student Performance Predictor

A Machine Learning web application that predicts whether a student is likely to **Pass or Fail** based on selected academic factors.

## 📌 Project Overview

The project uses **Logistic Regression** to predict student performance. The model is trained using the UCI Student Performance dataset and the application is built using Streamlit.

The prediction is based on:

* Study Time
* Previous Failures
* Absences
* First Period Grade (G1)
* Second Period Grade (G2)

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* Streamlit

## 📊 Dataset

The project uses the **UCI Student Performance dataset**.

The final grade (`G3`) is used to create the target:

* `G3 >= 10` → Pass
* `G3 < 10` → Fail

The `G3` value itself is not given to the model as an input feature.

## 🤖 Machine Learning Process

1. Load the student dataset
2. Create the Pass/Fail target
3. Select relevant features
4. Split the data into training and testing sets
5. Train a Logistic Regression model
6. Evaluate the model
7. Use the trained model in a Streamlit application

## 📈 Model Performance

The Logistic Regression model achieved approximately **90% test accuracy** on the 20% test split.

### Classification Report

| Class       | Precision | Recall | F1-Score |
| ----------- | --------: | -----: | -------: |
| Fail        |      0.81 |   0.93 |     0.86 |
| Pass        |      0.96 |   0.88 |     0.92 |
| **Overall** |           |        | **0.90** |

## 🌐 Web Application

The Streamlit application allows users to enter student details and receive:

* Predicted result: **Pass / Fail**
* Prediction probability
* Visual probability progress bar

## 📁 Project Structure

```text
student-performance-predictor/
│
├── app.py
├── train_model.py
├── student-mat.csv
├── requirements.txt
└── README.md
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## ⚠️ Limitations

The model's performance depends on the dataset and the selected features. The reported accuracy is based on a single train-test split and should not be interpreted as guaranteed performance on new real-world students.

## 🚀 Future Improvements

* Compare multiple Machine Learning algorithms
* Add data visualizations
* Add more student-related features
* Use cross-validation for more reliable evaluation
* Deploy the application online
