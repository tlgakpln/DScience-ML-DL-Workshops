from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_missing_data(data):
    missing_data = data.isnull().sum()
    print("\nMissing Data:")
    print(missing_data)


def analyze_outliers(data, feature_names=None):
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data[feature_names])
    plt.title("Outlier Analysis")
    plt.xticks(rotation=45)
    plt.show()


def analyze_class_distribution(target):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target)
    plt.title("Class Distribution")
    plt.show()


def analyze_dataset(data, target, feature_names=None, target_names=None):
    df = pd.DataFrame(data=data, columns=feature_names)
    df['target'] = target

    print("DataFrame Head:")
    print(df.head())

    print("\nDataFrame Info:")
    print(df.info())

    print("\nDataFrame Describe:")
    print(df.describe())

    if feature_names is not None:
        print("\nCorrelation Matrix:")
        correlation_matrix = df.corr()
        print(correlation_matrix)

        print("\nHistograms:")
        df.hist(bins=20, figsize=(10, 8))
        plt.show()

        print("\nOutlier Analysis:")
        analyze_outliers(df, feature_names)

    if target_names is not None:
        print("\nTarget Value Counts:")
        print(df['target'].value_counts())

        print("\nClass Distribution:")
        analyze_class_distribution(df['target'])


# You can analyze data

wine = datasets.load_wine()
analyze_dataset(wine.data, wine.target, wine.feature_names, wine.target_names)

cancer = datasets.load_breast_cancer()
analyze_dataset(cancer.data, cancer.target, cancer.feature_names, cancer.target_names)

