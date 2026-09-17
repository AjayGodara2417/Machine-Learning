# Titanic Survival Prediction --- Machine Learning

This repository contains my **Titanic Survival Prediction** machine
learning project, developed step-by-step in a Kaggle notebook.

The project starts with **Exploratory Data Analysis (EDA)** and
continues through **data preprocessing, feature engineering, multiple
classification models, model comparison, cross-validation, and
hyperparameter tuning**.

The goal is to predict whether a passenger survived the Titanic
disaster.

------------------------------------------------------------------------

## Project Overview

The Titanic dataset contains information about passengers such as:

-   Passenger class
-   Name
-   Sex
-   Age
-   Number of siblings/spouses aboard
-   Number of parents/children aboard
-   Ticket
-   Fare
-   Cabin
-   Port of embarkation

The target variable is:

``` text
Survived
0 → Did not survive
1 → Survived
```

Because the target contains two classes, this is a **binary
classification problem**.

------------------------------------------------------------------------

# Machine Learning Workflow

The project follows this workflow:

``` text
Load Dataset
     ↓
Understand Dataset
     ↓
Exploratory Data Analysis (EDA)
     ↓
Missing Value Analysis
     ↓
Duplicate Analysis
     ↓
Univariate Analysis
     ↓
Bivariate Analysis
     ↓
Multivariate Analysis
     ↓
Outlier Analysis
     ↓
Feature Engineering
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
KNN
     ↓
SVM
     ↓
Decision Tree
     ↓
Random Forest
     ↓
Naive Bayes
     ↓
Model Comparison
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
```

------------------------------------------------------------------------

# 1. Loading the Dataset

The Titanic dataset was loaded using Pandas:

``` python
import numpy as np
import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")
```

The first few records were inspected using:

``` python
df.head()
```

Additional rows were also inspected using:

``` python
df.tail()
df.sample(5)
```

------------------------------------------------------------------------

# 2. Understanding the Dataset

The shape of the dataset was checked using:

``` python
df.shape
```

The dataset contains:

``` text
891 rows
12 original columns
```

The columns were inspected using:

``` python
df.columns
```

Data types were checked using:

``` python
df.info()
df.dtypes
```

Numerical and categorical columns were identified using:

``` python
df.select_dtypes(include=np.number).columns
```

and:

``` python
df.select_dtypes(include="object").columns
```

------------------------------------------------------------------------

# 3. Missing Value Analysis

Missing values were checked using:

``` python
df.isnull().sum()
```

The main missing values were:

  Column         Missing Values
  ------------ ----------------
  `Age`                     177
  `Cabin`                   687
  `Embarked`                  2

The remaining original columns did not contain missing values.

------------------------------------------------------------------------

# 4. Duplicate Analysis

Duplicate rows were checked:

``` python
df.duplicated().sum()
```

The notebook found:

``` text
0 duplicate rows
```

Passenger ID duplication was also checked:

``` python
df["PassengerId"].duplicated().sum()
```

------------------------------------------------------------------------

# 5. Statistical Analysis

A general statistical summary was generated using:

``` python
df.describe()
```

The distribution of the `Age` feature was also examined:

``` python
df["Age"].describe()
```

This helped understand:

-   Mean
-   Standard deviation
-   Minimum
-   Quartiles
-   Maximum

------------------------------------------------------------------------

# 6. Target Variable Analysis

The target variable is:

``` python
Survived
```

The number of survivors and non-survivors was checked using:

``` python
df["Survived"].value_counts()
```

The dataset contained:

``` text
Did not survive → 549
Survived        → 342
```

A count plot was used to visualize the target:

``` python
sns.countplot(data=df, x="Survived")
plt.show()
```

The survival rate was approximately:

``` text
38.38% survived
61.62% did not survive
```

------------------------------------------------------------------------

# 7. Univariate Analysis

Individual features were analyzed to understand their distributions.

## Sex

``` python
df["Sex"].value_counts()
sns.countplot(data=df, x="Sex")
```

## Passenger Class

``` python
df["Pclass"].value_counts()
sns.countplot(data=df, x="Pclass")
```

## Embarkation Port

``` python
sns.countplot(data=df, x="Embarked")
```

## Age

``` python
sns.countplot(data=df, x="Age")
```

A histogram with KDE was also used:

``` python
sns.histplot(
    data=df,
    x="Age",
    kde=True
)
```

## Fare

``` python
sns.histplot(
    data=df,
    x="Fare",
    kde=True
)
```

## SibSp

``` python
sns.countplot(
    data=df,
    x="SibSp"
)
```

## Parch

``` python
sns.countplot(
    data=df,
    x="Parch"
)
```

------------------------------------------------------------------------

# 8. Outlier Analysis

Box plots were used to inspect possible outliers in:

### Age

``` python
sns.boxplot(
    data=df,
    x="Age"
)
plt.show()
```

### Fare

``` python
sns.boxplot(
    data=df,
    x="Fare"
)
plt.show()
```

The high-fare passengers were also inspected:

``` python
df[
    df["Fare"] > df["Fare"].quantile(0.95)
]
```

------------------------------------------------------------------------

# 9. Bivariate Analysis

The relationship between individual features and survival was explored.

## Sex vs Survival

``` python
pd.crosstab(
    df["Sex"],
    df["Survived"]
)
```

Survival percentages were calculated using:

``` python
pd.crosstab(
    df["Sex"],
    df["Survived"],
    normalize="index"
) * 100
```

Visualization:

``` python
sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)
```

The analysis showed substantial differences in survival proportions
between male and female passengers.

------------------------------------------------------------------------

## Passenger Class vs Survival

``` python
pd.crosstab(
    df["Pclass"],
    df["Survived"]
)
```

Visualization:

``` python
sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)
```

Survival rates by class were also calculated:

``` python
df.groupby("Pclass")["Survived"].mean()
```

The observed survival rates were approximately:

``` text
Pclass 1 → 62.96%
Pclass 2 → 47.28%
Pclass 3 → 24.24%
```

------------------------------------------------------------------------

## Embarked vs Survival

``` python
sns.countplot(
    data=df,
    x="Embarked",
    hue="Survived"
)
```

------------------------------------------------------------------------

## Age vs Survival

Box plots were used:

``` python
sns.boxplot(
    data=df,
    x="Survived",
    y="Age"
)
```

A distribution comparison was also made:

``` python
sns.histplot(
    data=df,
    x="Age",
    hue="Survived",
    kde=True,
    element="step"
)
```

------------------------------------------------------------------------

## Fare vs Survival

``` python
sns.boxplot(
    data=df,
    x="Survived",
    y="Fare"
)
```

------------------------------------------------------------------------

# 10. Correlation Analysis

Correlation between numerical variables was calculated using:

``` python
corr = df.select_dtypes(
    include=np.number
).corr()

corr
```

A heatmap was used:

``` python
plt.figure(figsize=(10, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

------------------------------------------------------------------------

# 11. Multivariate Analysis

Relationships involving multiple variables were explored.

For example:

``` python
sns.catplot(
    data=df,
    x="Pclass",
    hue="Survived",
    col="Sex",
    kind="count"
)

plt.show()
```

Another visualization:

``` python
sns.catplot(
    data=df,
    x="Pclass",
    y="Survived",
    hue="Sex",
    kind="bar"
)

plt.show()
```

These visualizations helped examine survival across combinations of
passenger class and sex.

------------------------------------------------------------------------

# 12. Feature Engineering

Feature engineering was used to create additional information from
existing columns.

## Family Size

A new feature was created:

``` python
df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)
```

The formula is:

``` text
FamilySize = SibSp + Parch + 1
```

The new feature was inspected:

``` python
df[
    ["SibSp", "Parch", "FamilySize"]
].head()
```

Its relationship with survival was also visualized:

``` python
sns.barplot(
    data=df,
    x="FamilySize",
    y="Survived"
)

plt.show()
```

------------------------------------------------------------------------

## IsAlone

A binary feature was created:

``` python
df["Alone"] = (
    df["FamilySize"] == 1
).astype(int)
```

Later, during preprocessing, the feature was named:

``` python
IsAlone
```

The survival relationship was analyzed using:

``` python
pd.crosstab(
    df["Alone"],
    df["Survived"],
    normalize="index"
) * 100
```

------------------------------------------------------------------------

## Title Extraction

Passenger titles were extracted from the `Name` column:

``` python
df["Title"] = df["Name"].str.extract(
    r",\s*([^.]*)\."
)
```

The titles were inspected:

``` python
df["Title"].value_counts()
```

------------------------------------------------------------------------

## Cabin Availability

Instead of directly using the highly-missing `Cabin` column, a binary
feature was created:

``` python
df["HasCabin"] = (
    df["Cabin"].notnull()
).astype(int)
```

This represents whether cabin information was available.

------------------------------------------------------------------------

## Additional Analysis

The number of unique tickets was checked:

``` python
df["Ticket"].nunique()
```

Ticket frequencies were also inspected:

``` python
df["Ticket"].value_counts().head(10)
```

Other relationships examined included:

``` python
df.groupby("Pclass")["Survived"].mean()
```

and:

``` python
df.groupby("Survived")[
    ["Age", "Fare", "SibSp", "Parch"]
].mean()
```

------------------------------------------------------------------------

# 13. Data Preprocessing

A copy of the DataFrame was created:

``` python
df_clean = df.copy()
```

The target and features were separated:

``` python
X = df_clean.drop(
    "Survived",
    axis=1
)

y = df_clean["Survived"]
```

------------------------------------------------------------------------

# 14. Dropping Unnecessary Features

The following columns were removed from the model features:

``` python
X = X.drop(
    [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ],
    axis=1
)
```

The reason was to avoid directly feeding
identifier/free-text/high-missing raw columns into the models.

Useful information from `Name` and `Cabin` had already been converted
into engineered features such as:

-   `Title`
-   `HasCabin`

------------------------------------------------------------------------

# 15. Grouping Rare Titles

Title frequency was calculated:

``` python
title_counts = df_clean["Title"].value_counts()
```

Titles occurring fewer than 10 times were identified:

``` python
rare_titles = title_counts[
    title_counts < 10
].index
```

They were grouped into:

``` text
Rare
```

using:

``` python
df_clean["Title"] = df_clean["Title"].replace(
    rare_titles,
    "Rare"
)
```

------------------------------------------------------------------------

# 16. Final Engineered Features

The final feature engineering included:

### FamilySize

``` python
df_clean["FamilySize"] = (
    df_clean["SibSp"] +
    df_clean["Parch"] +
    1
)
```

### IsAlone

``` python
df_clean["IsAlone"] = (
    df_clean["FamilySize"] == 1
).astype(int)
```

### HasCabin

``` python
df_clean["HasCabin"] = (
    df_clean["Cabin"].notna()
).astype(int)
```

------------------------------------------------------------------------

# 17. Train-Test Split

The dataset was divided into training and testing data:

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The final split was:

``` text
Training features: 712 rows
Testing features: 179 rows
```

`stratify=y` was used to maintain a similar class distribution in the
training and testing sets.

------------------------------------------------------------------------

# 18. Numerical and Categorical Features

The numerical features used were:

``` python
numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "HasCabin"
]
```

The categorical features were:

``` python
categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title"
]
```

------------------------------------------------------------------------

# 19. Numerical Preprocessing

Numerical features were processed using:

``` python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

The process is:

``` text
Numerical Features
       ↓
Median Imputation
       ↓
Standard Scaling
```

------------------------------------------------------------------------

# 20. Categorical Preprocessing

Categorical features were processed using:

``` python
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

The process is:

``` text
Categorical Features
       ↓
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

------------------------------------------------------------------------

# 21. ColumnTransformer

Both preprocessing workflows were combined:

``` python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
```

Overall:

``` text
                 X
                 │
        ┌────────┴────────┐
        ↓                 ↓
   Numerical          Categorical
        ↓                 ↓
 Median Impute      Most-Frequent Impute
        ↓                 ↓
     Scaling          One-Hot Encoding
        └────────┬────────┘
                 ↓
        Final Feature Matrix
```

The notebook also demonstrated the difference between:

``` python
preprocessor.fit_transform(X_train)
```

and:

``` python
preprocessor.transform(X_test)
```

The preprocessing is fitted on the training data and then applied to the
test data.

------------------------------------------------------------------------

# 22. Logistic Regression

Logistic Regression was used as the first classification model.

``` python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(
    max_iter=1000
)
```

The model was combined with the preprocessing pipeline:

``` python
from sklearn.pipeline import Pipeline

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)
```

The model was trained:

``` python
model.fit(
    X_train,
    y_train
)
```

Predictions:

``` python
y_pred = model.predict(X_test)
```

Prediction probabilities:

``` python
y_prob = model.predict_proba(
    X_test
)[:, 1]
```

------------------------------------------------------------------------

# 23. Logistic Regression Evaluation

## Accuracy

``` python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)
```

Test accuracy:

``` text
83.24%
```

------------------------------------------------------------------------

## Confusion Matrix

``` python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

The resulting confusion matrix was:

``` text
[[97 13]
 [17 52]]
```

A heatmap was also created to visualize it.

------------------------------------------------------------------------

## Classification Report

``` python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

The observed metrics for the positive class (`Survived = 1`) were
approximately:

``` text
Precision → 0.80
Recall    → 0.75
F1 Score  → 0.78
```

------------------------------------------------------------------------

## ROC Curve

The ROC curve was generated using:

``` python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
```

and plotted against a random-guess baseline.

------------------------------------------------------------------------

## ROC-AUC

``` python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(
    y_test,
    y_prob
)
```

Observed Logistic Regression ROC-AUC:

``` text
0.8715
```

------------------------------------------------------------------------

# 24. Decision Tree

A Decision Tree classifier was also tested:

``` python
from sklearn.tree import DecisionTreeClassifier

tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)
```

Observed test accuracy:

``` text
0.7877
```

------------------------------------------------------------------------

# 25. K-Nearest Neighbors (KNN)

KNN was tested using:

``` python
from sklearn.neighbors import KNeighborsClassifier

knn_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)
```

Observed initial test accuracy:

``` text
0.8045
```

Initial ROC-AUC:

``` text
0.8379
```

A range of K values was also tested:

``` python
k_values = [
    3, 5, 7, 9, 11, 15, 20
]
```

Each K value was evaluated using 5-fold cross-validation:

``` python
for k in k_values:

    knn = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=k
                )
            )
        ]
    )

    scores = cross_val_score(
        knn,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(
        f"K={k}: "
        f"{scores.mean():.4f}"
    )
```

------------------------------------------------------------------------

# 26. Support Vector Machine

An SVM model was tested using an RBF kernel:

``` python
from sklearn.svm import SVC

svm_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ]
)
```

Observed test results:

``` text
Accuracy → 0.8436
ROC-AUC  → 0.8440
```

------------------------------------------------------------------------

# 27. Random Forest

A Random Forest classifier was tested:

``` python
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)
```

Observed test results:

``` text
Accuracy → 0.8045
ROC-AUC  → 0.8262
```

------------------------------------------------------------------------

# 28. Naive Bayes

Gaussian Naive Bayes was also tested.

Because the standard OneHotEncoder output is sparse, a dense output was
used for GaussianNB:

``` python
categorical_transformer_nb = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)
```

A separate preprocessing configuration was created and combined with:

``` python
from sklearn.naive_bayes import GaussianNB

nb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_nb
        ),
        (
            "classifier",
            GaussianNB()
        )
    ]
)
```

Observed test results:

``` text
Accuracy → 0.8045
ROC-AUC  → 0.8490
```

------------------------------------------------------------------------

# 29. Model Comparison

The notebook compared the classification models using:

-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC-AUC

The test-set comparison was:

  Model                   Accuracy   Precision   Recall       F1   ROC-AUC
  --------------------- ---------- ----------- -------- -------- ---------
  Logistic Regression       0.8324      0.8000   0.7536   0.7761    0.8715
  KNN                       0.8045      0.7576   0.7246   0.7407    0.8379
  SVM                       0.8436      0.8154   0.7681   0.7910    0.8440
  Random Forest             0.8045      0.7500   0.7391   0.7445    0.8262
  Naive Bayes               0.8045      0.7297   0.7826   0.7552    0.8490

These values come from the notebook's test-set evaluation.

------------------------------------------------------------------------

# 30. Cross-Validation

5-fold cross-validation was used to compare model stability.

``` python
from sklearn.model_selection import cross_validate

scores = cross_validate(
    current_model,
    X_train,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
)
```

The average 5-fold results were:

  Model                   Accuracy   Precision   Recall       F1   ROC-AUC
  --------------------- ---------- ----------- -------- -------- ---------
  Logistic Regression       0.8287      0.7900   0.7545   0.7714    0.8682
  KNN                       0.8077      0.7736   0.7036   0.7360    0.8474
  SVM                       0.8189      0.7865   0.7255   0.7535    0.8506
  Random Forest             0.7937      0.7385   0.7182   0.7281    0.8550
  Naive Bayes               0.8020      0.7205   0.7949   0.7554    0.8516

Cross-validation was performed on the training data, keeping the test
set separate.

------------------------------------------------------------------------

# 31. Hyperparameter Tuning --- Logistic Regression

`GridSearchCV` was used to tune Logistic Regression.

``` python
from sklearn.model_selection import GridSearchCV

logistic_param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],
    "classifier__solver": [
        "liblinear",
        "lbfgs"
    ]
}
```

The grid search:

``` python
logistic_grid = GridSearchCV(
    estimator=model,
    param_grid=logistic_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

was fitted using:

``` python
logistic_grid.fit(
    X_train,
    y_train
)
```

The best parameters found were:

``` text
C = 1
solver = lbfgs
```

Best cross-validation accuracy:

``` text
0.8287
```

The tuned Logistic Regression achieved on the held-out test set:

``` text
Test Accuracy → 0.8324
Test ROC-AUC  → 0.8715
```

------------------------------------------------------------------------

# 32. Hyperparameter Tuning --- KNN

KNN was tuned using:

``` python
knn_param_grid = {
    "classifier__n_neighbors": [
        3, 5, 7, 9, 11, 15, 21
    ],
    "classifier__weights": [
        "uniform",
        "distance"
    ],
    "classifier__p": [
        1,
        2
    ]
}
```

Grid search:

``` python
knn_grid = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

The search was fitted using:

``` python
knn_grid.fit(
    X_train,
    y_train
)
```

The best parameters found were:

``` text
n_neighbors = 3
weights      = uniform
p            = 1
```

Best cross-validation accuracy:

``` text
0.8329
```

The tuned KNN achieved:

``` text
Test Accuracy → 0.8101
Test ROC-AUC  → 0.8598
```

------------------------------------------------------------------------

# 33. Tuned Model Comparison

The final tuned-model comparison from the notebook was:

  Model                         CV Accuracy   Test Accuracy   Test ROC-AUC
  --------------------------- ------------- --------------- --------------
  Tuned Logistic Regression          0.8287          0.8324         0.8715
  Tuned KNN                          0.8329          0.8101         0.8598

This table is used to understand how the tuned models performed on the
training cross-validation process and the previously unseen test set.

------------------------------------------------------------------------

# 34. Key Machine Learning Concepts Learned

This project covers:

-   Exploratory Data Analysis
-   Data types
-   Missing-value analysis
-   Duplicate analysis
-   Statistical analysis
-   Target analysis
-   Univariate analysis
-   Bivariate analysis
-   Multivariate analysis
-   Correlation analysis
-   Outlier analysis
-   Feature engineering
-   Train/test splitting
-   Stratified splitting
-   Numerical imputation
-   Categorical imputation
-   StandardScaler
-   One-Hot Encoding
-   Pipeline
-   ColumnTransformer
-   Logistic Regression
-   KNN
-   Decision Tree
-   Support Vector Machine
-   Random Forest
-   Naive Bayes
-   Classification metrics
-   Confusion Matrix
-   Precision
-   Recall
-   F1 Score
-   ROC Curve
-   ROC-AUC
-   Cross-validation
-   GridSearchCV
-   Hyperparameter tuning

------------------------------------------------------------------------

# 35. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Kaggle Notebook
-   GitHub

------------------------------------------------------------------------

# 36. Repository Structure

A suggested repository structure:

``` text
titanic-machine-learning/
│
├── EDA, Logistic regression.ipynb
├── README.md
├── Titanic-Dataset.csv
└── requirements.txt
```

Example `requirements.txt`:

``` text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

------------------------------------------------------------------------

# 37. Future Improvements

Possible next steps for this project include:

-   More systematic feature engineering
-   Feature selection
-   Logistic Regression regularization tuning
-   More extensive KNN tuning
-   SVM hyperparameter tuning
-   Random Forest hyperparameter tuning
-   Threshold analysis
-   Calibration of predicted probabilities
-   Additional ensemble models
-   More detailed error analysis
-   Final model interpretation

------------------------------------------------------------------------

# Conclusion

This project demonstrates a complete machine learning workflow on the
Titanic dataset, beginning with exploratory data analysis and
progressing through preprocessing, feature engineering, classification
models, evaluation, cross-validation, and hyperparameter tuning.

The project also demonstrates why preprocessing should be integrated
into a machine learning `Pipeline`: numerical and categorical features
require different transformations, and fitting preprocessing only on
training data helps avoid data leakage.

The next stage of the broader machine learning learning path is to apply
the same workflow to a **regression problem**, such as the California
Housing dataset, where Linear Regression can be used appropriately.
