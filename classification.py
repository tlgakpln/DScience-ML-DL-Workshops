import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class WineClassificationPipeline:
    def __init__(self, models):
        self.models = models

    def preprocess_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_accuracy = grid_search.best_score_
        return best_model, best_accuracy

    def feature_importance_plot(self, model, feature_names):
        feature_importances = model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_feature_names = np.array(feature_names)[sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_names)), feature_importances[sorted_idx], align="center")
        plt.xticks(range(len(feature_names)), sorted_feature_names, rotation=45, ha="right")
        plt.xlabel("Feature")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    def feature_engineering(self, X):
        X_new = X.copy()

        low_group = ['alcohol', 'malic_acid', 'ash']
        medium_group = ['alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids']
        high_group = ['nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                      'od280/od315_of_diluted_wines', 'proline']

        X_new['mean_low_group'] = X_new[low_group].mean(axis=1)
        X_new['mean_medium_group'] = X_new[medium_group].mean(axis=1)
        X_new['mean_high_group'] = X_new[high_group].mean(axis=1)

        cluster_features = ['total_phenols', 'flavanoids', 'color_intensity', 'hue', 'alcalinity_of_ash']
        X_new['cluster_features'] = X_new[cluster_features].sum(axis=1)

        X_new['high_alcohol'] = (X_new['alcohol'] > 13.5).astype(int)
        X_new['high_color_intensity'] = (X_new['color_intensity'] > 5.0).astype(int)

        return X_new

    def run_pipeline(self, X, y, feature_names):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.preprocess_data(X_train)
        X_test_scaled = self.preprocess_data(X_test)

        best_model = None
        best_accuracy = 0.0

        for model_name, model, param_grid in self.models:
            print(f"Evaluating {model_name}...")

            accuracy = self.evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
            print(f"{model_name} Accuracy: {accuracy:.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

        print(f"Best Model: {best_model} with Accuracy: {best_accuracy:.2f}")

        for model_name, model, param_grid in self.models:
            if model_name == best_model:
                print(f"Performing Hyperparameter Tuning for {model_name}...")
                best_model, best_accuracy = self.hyperparameter_tuning(model, param_grid, X_train_scaled, y_train)
                print(f"Best Model after Hyperparameter Tuning for {model_name}: {best_model}")
                print(f"Best Accuracy after Hyperparameter Tuning: {best_accuracy:.2f}")

                print(f"Plotting Feature Importance for {model_name}...")
                best_model.fit(X_train_scaled, y_train)
                self.feature_importance_plot(best_model, feature_names)


wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

models = [
    ("RandomForest", RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200, 1000],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    ("SVM", SVC(random_state=42), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))
    }),
    ("LogisticRegression", LogisticRegression(random_state=42), {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42), {
        'n_estimators': [50, 100, 200, 1000],
        'learning_rate': [0.001, 0.01, 0.08, 0.1],
        'max_depth': [3, 4, 5]
    }),
    ("AdaBoost", AdaBoostClassifier(random_state=42), {
        'n_estimators': [50, 100, 200, 1000],
        'learning_rate': [0.001, 0.01, 0.08, 0.1],
    })
]

pipeline = WineClassificationPipeline(models)
pipeline.run_pipeline(X, y, feature_names)
