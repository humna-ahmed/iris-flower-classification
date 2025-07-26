# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the Iris dataset from a specific path"""
    file_path = r"C:\Users\Zain\OneDrive - Higher Education Commission\Desktop\iris.csv"
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data: handle missing values, feature selection, etc."""
    # Check for missing values
    if data.isnull().sum().any():
        print("Missing values found. Handling them...")
        data = data.dropna()  # or use data.fillna(data.mean())
    
    # Check if 'species' column exists (case insensitive)
    if not any(col.lower() == 'species' for col in data.columns):
        raise ValueError("The dataset must contain a 'species' column for classification.")
    
    # Standardize column names (in case they have different capitalization)
    data.columns = data.columns.str.lower()
    
    # Separate features and target
    X = data.drop('species', axis=1)
    y = data['species']
    
    return X, y

def train_model(X_train, y_train):
    """Train a Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance"""
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    # Load data
    print("Loading data from your specified path...")
    data = load_data()
    if data is None:
        return
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Preprocess data
    try:
        X, y = preprocess_data(data)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\nTraining the model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': data.drop('species', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.show()

if __name__ == "__main__":
    main()