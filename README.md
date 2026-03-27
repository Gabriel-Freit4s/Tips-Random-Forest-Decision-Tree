# Tips Prediction - Decision Trees and Random Forest

This project involves exploratory data analysis and machine learning modeling on a restaurant tips dataset (`tips.csv`). The main objective is to predict whether a tip will be proportionally "high" or "low" relative to the total bill amount, using **Decision Tree** and **Random Forest** algorithms.

## 1. Project Overview and Dataset

The dataset used is **"waiter-tip-prediction"** from Kaggle, which contains 244 records and 7 columns, detailing customer information in a restaurant.

The variables present in the original dataset include:
- **total_bill:** Total bill amount
- **tip:** Tip amount
- **sex:** Payer's gender (Male/Female)
- **smoker:** Whether there were smokers at the table (Yes/No)
- **day:** Day of the week (Thur, Fri, Sat, Sun)
- **time:** Meal period (Lunch/Dinner)
- **size:** Number of people at the table

### Pre-processing (Feature Engineering)
The data preparation step consisted of:
1. **Null Value Verification:** Confirmed that there are no null values in the DataFrame.
2. **Dummy Variable Creation:** Transformation of categorical variables (`sex`, `smoker`, `day`, `time`) into numerical (0 or 1).
3. **Target Variable Creation (`tip_total_bill`):**
   - The tip proportion relative to the total bill (`tip` / `total_bill`) was calculated.
   - The median of this proportion was obtained to divide the dataset into two classes (50% / 50%).
   - A binary variable `tip_total_bill` was created, where `1` indicates tips above the median and `0` indicates tips below or equal to the median. The original `total_bill` and `tip` variables were dropped from the final model.

---

## 2. Applied Models

The data was split into 70% for training and 30% for testing, using stratification on the target variable to maintain class proportions.

### 2.1 Decision Tree
- The model was tested with different criteria (`entropy`, `gini`, and `log_loss`), yielding similar results.
- An analysis was performed by varying the maximum tree depth (`max_depth` from 1 to 10).
- The best result was obtained with `max_depth = 2`, achieving an accuracy of approximately **55.4%**. Increasing the depth beyond this led to a drop in performance on the test set, indicating strong *overfitting* (training accuracy was around 66.4%, while test accuracy dropped to ~44.5% with greater depths).

### 2.2 Random Forest
- The algorithm was tested by varying the number of estimators (`n_estimators`: 10, 100, 1000).
- Different criteria (`gini`, `entropy`, `log_loss`) were also tested.
- A search was conducted combining the variation of `max_depth` (1 to 10) with the number of estimators.
- The best Random Forest model was found with **`n_estimators = 10` and `max_depth = 8`**, achieving an accuracy of **59.4%** on the test set.

---

## 3. Validation and Conclusions

Various validation techniques were used to compare the models:
- **Classification Report and Confusion Matrix:** Showed relatively low precision and recall (around 44% to 45% in less optimized models), highlighting the model's difficulty in correctly separating classes.
- **Cross-Validation:** Compared the mean and standard deviation of accuracy between the Decision Tree and Random Forest.
- **ROC Curve and AUC:** Plotted to evaluate the discrimination capacity between classes for each model.
- **Learning Curves:** Clearly showed the gap between training performance (which approached 100% in Random Forest) and test performance (which stagnated below 60%), confirming **Overfitting**.
- **Feature Importance:** Graphs demonstrated which variables were most decisive for each model's decisions.

### Final Conclusion

1. **Regarding the dataset:** The `tips.csv` dataset is quite small (only 244 records). This results in a dataset with **high variability**. Furthermore, the variables are categorical with few options, which limits the algorithms' ability to find more complex and consistent separations, making them highly prone to *overfitting*.
2. **Regarding modeling:** Even with proper data treatment, the results proved unstable. Random Forest slightly outperformed the simple Decision Tree (59.4% vs 55.4%), but both suffered from *overfitting*.
3. **Suggested next steps:** To improve predictive performance in this problem, it would be necessary to collect **more data**, introduce **new explanatory variables** (better features), and apply more robust hyperparameter optimization techniques (such as a complete `GridSearchCV`).

