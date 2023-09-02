import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(data=np.c_[X, y], columns=np.append(data.feature_names, ['target']))

print("Missing Values:")
print(df.isnull().sum())

missing_data_columns = df.columns[df.isnull().any()].tolist()

for column in missing_data_columns:
    correlated_columns = df.columns[df.corr().abs()[column] > 0.7].tolist()
    for correlated_column in correlated_columns:
        df[column].fillna(df.groupby(correlated_column)[column].transform("mean"), inplace=True)

# Check if missing values are filled
print("\nData with Missing Values Filled:")
print(df.isnull().sum())

# added new features
df['mean_radius_times_texture'] = df['mean radius'] * df['mean texture']
df['mean_radius_plus_texture'] = df['mean radius'] + df['mean texture']
correlation_matrix = df.corr()

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }),
    "Logistic Regression": (LogisticRegression(), {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }),
    "Support Vector Classifier": (SVC(), {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }),
    "Extra Trees": (ExtraTreesClassifier(), {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(), {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    "LightGBM": (LGBMClassifier(), {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 1],
        'max_depth': [-1, 10, 20, 30],
        'num_leaves': [31, 63, 127]
    }),
    "CatBoost": (CatBoostClassifier(silent=True), {
        'iterations': [100, 300, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 1],
        'depth': [4, 6, 8, 10]
    })
}

# train models and evaluate
results = {}
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    results[model_name] = {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy
    }

# Results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Accuracy on Test Data: {result['accuracy']:.4f}")
    print()
