import io
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_confusion_matrix(model, dataset_path: str, target_column: str):
    """Generates a confusion matrix plot for a classification model."""
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # The model passed here is a pipeline (Preprocessor + Estimator)
    # We need to replicate the exact same split as training to be fair,
    # or use the full dataset but that's cheating.
    # Let's assume the standard split.
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)

    # Get classes from the estimator step of the pipeline
    estimator = model.named_steps['model'] if 'model' in model.named_steps else model

    # If classes_ attribute exists (it should for classifiers)
    classes = getattr(estimator, 'classes_', None)

    cm = confusion_matrix(y_test, preds, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_cluster_scatter_plot(model, dataset_path: str):
    """Generates a 2D scatter plot for clustering results using PCA."""
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)

    # Model is a Pipeline
    preprocessor = model.named_steps['preprocessor']
    estimator = model.named_steps['model']

    X_processed = preprocessor.transform(df)

    # If sparse (due to OneHotEncoder), convert to dense for PCA
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Use PCA to reduce to 2 dimensions for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    # If the model is already fitted (it is), we can just predict.
    # For clustering like KMeans, .predict() works. For DBSCAN, we need fit_predict or .labels_ if we had the object
    # But here we are loading a fresh joblib object? No, we load the saved one.
    # Sklearn clustering models usually store labels_

    if hasattr(estimator, 'labels_'):
        labels = estimator.labels_
    elif hasattr(estimator, 'predict'):
        labels = estimator.predict(X_processed)
    else:
        labels = estimator.fit_predict(X_processed)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7)
    plt.title('Clustering Results (PCA-reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_feature_importance_plot(model, dataset_path: str, target_column: str):
    """Generates a feature importance plot for tree-based models."""
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
    X = df.drop(columns=[target_column])

    # Access pipeline steps
    estimator = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']

    if not hasattr(estimator, 'feature_importances_'):
        return None

    importances = estimator.feature_importances_

    # Get feature names from preprocessor
    feature_names = []

    # Numeric features (passed through scaler)
    # We need to know which transformers were used.
    # In our code we have 'num' and 'cat'

    try:
        # This is available in scikit-learn >= 1.0
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback if version issues, though we installed latest
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Create DataFrame for plotting
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20) # Top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title('Feature Importance')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_actual_vs_predicted_plot(model, dataset_path: str, target_column: str):
    """Generates an Actual vs Predicted scatter plot for regression models."""
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.7)

    # Plot diagonal line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf
