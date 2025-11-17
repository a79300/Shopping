"""
Shopping Predictor - Machine Learning model to predict shopping behavior
Uses K-Nearest Neighbors to classify whether a user will make a purchase
based on their browsing session data.
"""

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Percentage of data to use for testing (40% test, 60% train)
TEST_SIZE = 0.4


def main():
    """
    Main function to run the shopping predictor model.
    Loads data, trains model, makes predictions, and evaluates performance.
    """

    # Check command-line arguments - expect exactly one argument (CSV filename)
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    # evidence = list of feature vectors, labels = list of outcomes (0 or 1)
    evidence, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets using sklearn's train_test_split
    # X_train/y_train = data for training, X_test/y_test = data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train the K-NN model on training data and make predictions on test data
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Evaluate model performance using sensitivity and specificity metrics
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results showing accuracy and performance metrics
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file and prepare it for ML model.
    
    Args:
        filename: Path to CSV file containing shopping session data
        
    Returns:
        tuple: (evidence, labels) where evidence is a list of feature vectors
               and labels is a list of binary outcomes (1=purchase, 0=no purchase)
    """
    
    # Initialize lists to store features and labels
    evidence = []
    labels = []

    # Dictionary to convert month names to numerical values (0-11)
    months = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }

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
                float(row["Informational_Duration"]),
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


def train_model(evidence, labels):
    """
    Train a K-Nearest Neighbors classifier on the provided data.
    
    Args:
        evidence: List of feature vectors (training data)
        labels: List of corresponding labels (0 or 1)
        
    Returns:
        Trained KNeighborsClassifier model
    """
    
    # Initialize K-NN classifier with k=1 (classify based on nearest neighbor)
    model = KNeighborsClassifier(n_neighbors=1)
    
    # Alternative: Random Forest (uncomment to use instead of K-NN)
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the evidence and labels
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
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
    true_positives = 0  # Correctly predicted purchases
    true_negatives = 0  # Correctly predicted non-purchases
    
    # Count total actual positives and negatives
    total_positives = labels.count(1)  # Total actual purchases
    total_negatives = labels.count(0)  # Total actual non-purchases

    # Compare each prediction to actual label
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            # Model correctly predicted a purchase
            true_positives += 1
        elif actual == 0 and predicted == 0:
            # Model correctly predicted no purchase
            true_negatives += 1

    # Calculate sensitivity (recall for positive class)
    # Avoid division by zero if no positive examples exist
    sensitivity = true_positives / total_positives if total_positives > 0 else 0
    
    # Calculate specificity (recall for negative class)
    # Avoid division by zero if no negative examples exist
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
