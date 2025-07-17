# Lung Cancer Prediction using Logistic Regression

This project aims to predict the likelihood of lung cancer using survey data. We perform data preprocessing, visualization, and model training using logistic regression.

## 📁 Dataset

- **Source:** "https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia"
- Contains responses to various lifestyle and health-related questions, along with a label indicating lung cancer status.

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## 🚀 Steps Performed

1. **Data Loading**  
   Load the CSV dataset using Pandas.

2. **Data Exploration**  
   - View structure and summary statistics.
   - Check for missing values.

3. **Data Preprocessing**  
   - Encode categorical variables (`GENDER`, `LUNG_CANCER`).
   - Convert age to integers.

4. **Visualization**  
   - Class balance using `countplot`.
   - Feature correlation using `heatmap`.

5. **Model Preparation**  
   - Split data into features (`X`) and label (`y`).
   - Train-test split (80/20).

6. **Model Training**  
   - Logistic Regression using Scikit-learn.

7. **Model Evaluation**  
   - Accuracy
   - Confusion matrix
   - Precision, Recall, F1-score

## 📊 Results

- **Model Accuracy:** `XX%` (replace with actual output)
- Confusion Matrix and classification report give deeper insights into performance.

## 📌 Conclusion

The model helps predict lung cancer presence based on survey inputs. With proper medical validation, such models could support early warning systems.

## 📎 Future Work

- Try different models (Random Forest, SVM, etc.)
- Tune hyperparameters
- Use real clinical datasets
