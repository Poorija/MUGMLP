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
    X = df.drop(columns=[target_column]).select_dtypes(include=['number'])
    y = df[target_column]

    # Simple train/test split to get predictions
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
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
    X = df.select_dtypes(include=['number'])
    X_scaled = StandardScaler().fit_transform(X)

    # Use PCA to reduce to 2 dimensions for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    labels = model.fit_predict(X_scaled)

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
