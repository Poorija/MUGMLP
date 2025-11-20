import pandas as pd
import joblib
import json
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import crud

# Scikit-learn & XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Model Registries ---

TASK_REGISTRY = {
    "classification": {
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": xgb.XGBClassifier,
        "SimpleNN": "pytorch", # Special key for PyTorch models
    },
    "clustering": {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
    },
    "dimensionality_reduction": {
        "PCA": PCA,
    }
}

# --- PyTorch Implementation (remains the same) ---
class SimpleNN(nn.Module):
    # ... (code from previous step)
    pass
def train_pytorch_model(df, target_column, model_params, model_id):
    # ... (code from previous step)
    pass

# --- Main Training Task ---

def train_model_task(model_id: int, dataset_id: int, model_info: dict):
    db: Session = SessionLocal()
    try:
        crud.update_model_status(db, model_id, "running")

        dataset = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset.filename}"
        df = pd.read_csv(file_location) if dataset.filename.endswith('.csv') else pd.read_excel(file_location)

        model_type = model_info.get('model_type')
        task_type = model_info.get('task_type') # e.g., 'classification', 'clustering'
        hyperparams = model_info.get('hyperparameters', {})
        target_column = model_info.get('target_column')

        # Find the model class from the registry
        model_class = TASK_REGISTRY.get(task_type, {}).get(model_type)
        if not model_class:
            raise ValueError(f"Model type '{model_type}' not found for task '{task_type}'")

        # --- Data Preparation ---
        # For supervised tasks, separate X and y. For unsupervised, use all numeric data.
        if task_type == "classification":
            if not target_column:
                raise ValueError("Target column is required for classification")
            X = df.drop(columns=[target_column]).select_dtypes(include=['number'])
            y = df[target_column]
            # Encode labels for classification
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        else:
            # For clustering/PCA, use all numeric data
            X = df.select_dtypes(include=['number'])
            # Scale data for distance-based algorithms
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        # --- Model Training & Evaluation ---
        metrics = {}
        if model_class == "pytorch": # Handling PyTorch case
            metrics, model_path = train_pytorch_model(df, target_column, hyperparams, model_id)
        else:
            model = model_class(**hyperparams)

            if task_type == "classification":
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, preds, average='weighted', zero_division=0),
                }
            elif task_type == "clustering":
                labels = model.fit_predict(X_scaled)
                # Avoid silhouette score for DBSCAN if only one cluster is found (-1 label for noise)
                if model_type == 'DBSCAN' and len(set(labels)) <= 2:
                     metrics = {"clusters_found": len(set(labels)) - (1 if -1 in labels else 0)}
                else:
                    metrics = {
                        "silhouette_score": silhouette_score(X_scaled, labels),
                        "clusters_found": len(set(labels))
                    }
            elif task_type == "dimensionality_reduction":
                model.fit(X_scaled)
                metrics = {
                    "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                    "n_components": model.n_components_
                }

            # Save the Scikit-learn/XGBoost model
            model_path = f"ml_models/model_{model_id}.joblib"
            joblib.dump(model, model_path)

        crud.update_model_status(db, model_id, "completed", metrics=metrics, model_path=model_path)

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        print(f"Training failed for model_id {model_id}: {e}")

    finally:
        db.close()
