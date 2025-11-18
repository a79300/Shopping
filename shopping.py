"""
Shopping Predictor - Machine Learning model to predict shopping behavior
Uses K-Nearest Neighbors to classify whether a user will make a purchase
based on their browsing session data.
"""

import csv
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Percentage of data to use for testing (40% test, 60% train)
TEST_SIZE = 0.4

def main():
    """
    Main function to run the shopping predictor model.
    Loads data, trains model, makes predictions, and evaluates performance.
    """

    # Check command-line arguments - expect exactly one argument (CSV filename)
    if len(sys.argv) != 2:
        sys.exit("Uso: python shopping.py data.csv")

    # Load data from spreadsheet and split into train and test sets
    # evidence = list of feature vectors, labels = list of outcomes (0 or 1)
    evidence, labels = load_data(sys.argv[1])

    # Split data into training and testing sets using sklearn's train_test_split
    # X_train/y_train = data for training, X_test/y_test = data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # models
    models = {
        # Train the K-NN model on training data and make predictions on test data
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga"),
        "SVM": SVC(kernel="linear"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42, early_stopping=True)
    }

    # Tuning for Random Forest
    param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)
    models["Random Forest (Tuned)"] = grid.best_estimator_

    # Evaluate models
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        sensitivity, specificity = evaluate(y_test, predictions)
        accuracy = (y_test == predictions).sum() / len(y_test)
        correct = (y_test == predictions).sum()
        incorrect = (y_test != predictions).sum()
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Corrects": correct,
            "Incorrects": incorrect,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

    # Create table
    df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n=== Comparison of the Models ===\n")
    print(df.to_string(index=False))

    # save CSV
    df.to_csv("results_models.csv", index=False)

def load_data(filename):
    evidence = []
    labels = []
    months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
              "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

    # Open and read the CSV file
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        # Process each row of shopping session data
        for row in reader:
            # Create a feature vector with 17 features for each session
            sample = [
                # Administrative pages visited and time spent
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                # Informational pages visited and time spent
                int(row["Informational"]),
                # Product-related pages visited and time spent
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                # Bounce and exit rates (behavior metrics)
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                # Page value and special day proximity (0-1)
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                # Month converted to integer (0-11)
                months[row["Month"]],
                # Technical details: OS, browser, region, traffic source
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                # Visitor type: 1 for returning, 0 for new
                1 if row["VisitorType"].strip() == "Returning_Visitor" else 0,
                # Weekend visit: 1 for yes, 0 for no
                1 if row["Weekend"].strip().lower() == "true" else 0,
            ]
            evidence.append(sample)

            # Label: 1 if purchase was made (Revenue=TRUE), 0 otherwise
            labels.append(1 if row["Revenue"].strip().lower() == "true" else 0)

    return (evidence, labels)

def evaluate(labels, predictions):
    """
    true_negatives = sum(1 for a, p in zip(labels, predictions) if a == 0 and p == 0)
    Evaluate model performance by calculating sensitivity and specificity.
    
    Args:
        labels: List of actual labels (ground truth)
        predictions: List of predicted labels from the model
        
    Returns:
        tuple: (sensitivity, specificity)
            - sensitivity: True Positive Rate (proportion of actual buyers correctly identified)
            - specificity: True Negative Rate (proportion of non-buyers correctly identified)
    """
    
    # Initialize counters for correct predictions
    true_positives = sum(1 for a, p in zip(labels, predictions) if a == 1 and p == 1)  # Correctly predicted purchases
    true_negatives = sum(1 for a, p in zip(labels, predictions) if a == 0 and p == 0)  # Correctly predicted non-purchases
    
    # Count total actual positives and negatives
    total_positives = labels.count(1)  # Total actual purchases
    total_negatives = labels.count(0)  # Total actual non-purchases

    # Calculate sensitivity (recall for positive class)
    # Avoid division by zero if no positive examples exist
    sensitivity = true_positives / total_positives if total_positives > 0 else 0

    # Calculate specificity (recall for negative class)
    # Avoid division by zero if no negative examples exist
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()