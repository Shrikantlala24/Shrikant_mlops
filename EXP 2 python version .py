
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.head()

iris.isnull().sum()

iris[iris.duplicated()]

iris.drop_duplicates(inplace = True)
iris.duplicated().sum()

sns.kdeplot(
    data=iris,
    x="petal_length",
    y="petal_width",
    hue="species",
    fill=True
)

features = iris.columns[:-1]  # exclude species

for feature in features:
    plt.figure(figsize=(6,4))

    sns.kdeplot(
        data=iris,
        x=feature,
        hue="species",
        fill=False,      # LINE graph (important)
        linewidth=2
    )

    plt.title(f"PDF (KDE) of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Assume last column is target
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))