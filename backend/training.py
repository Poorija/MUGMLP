import pandas as pd
import joblib
import json
import asyncio
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import crud
from .websocket_manager import manager
from .training_vision import train_vision_model_task
from .training_mamba import train_mamba_model_task
from .training_dpo import train_dpo_task
from .data_tools import generate_synthetic_data_task, llm_judge_task
from .training_moe import train_moe_model_task

# Scikit-learn & XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, silhouette_score,
    mean_squared_error, r2_score, mean_absolute_error
)
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
        "Auto": "auto", # Special key for AutoML
    },
    "regression": {
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "XGBRegressor": xgb.XGBRegressor,
        "SVR": SVR,
        "SimpleNN": "pytorch",
        "Auto": "auto", # Special key for AutoML
    },
    "image_classification": {
        "ViT": "vision_transformer"
    },
    "text_generation": {
        "Mamba": "mamba_ssm",
        "MoE": "mixture_of_experts"
    },
    "alignment": {
        "DPO": "direct_preference_optimization"
    },
    "data_generation": {
        "SyntheticData": "synthetic_data_generation"
    },
    "evaluation": {
        "LLMJudge": "llm_as_a_judge"
    },
    "clustering": {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
    },
    "dimensionality_reduction": {
        "PCA": PCA,
    }
}

# --- PyTorch Implementation ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, task_type="classification"):
        super(SimpleNN, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        if task_type == "classification":
            # For classification (multi-class), we usually use CrossEntropyLoss which takes logits
            # but we can apply Softmax for inference. The output here is logits.
            pass
        elif task_type == "regression":
            # For regression, usually linear output
            pass

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_pytorch_model(df, target_column, model_params, model_id, task_type="classification"):
    input_dim = df.shape[1] - 1 # Exclude target

    if task_type == "classification":
        output_dim = len(df[target_column].unique())
        criterion = nn.CrossEntropyLoss()
    else: # Regression
        output_dim = 1
        criterion = nn.MSELoss()

    hidden_layers = [int(x) for x in str(model_params.get('hidden_layers', '64,32')).split(',')]
    epochs = int(model_params.get('epochs', 10))
    lr = float(model_params.get('learning_rate', 0.001))

    model = SimpleNN(input_dim, hidden_layers, output_dim, task_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Data Prep
    X = df.drop(columns=[target_column]).select_dtypes(include=['number']).values
    y = df[target_column].values

    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Broadcast progress
        progress_msg = json.dumps({
            "model_id": model_id,
            "status": "running",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "loss": running_loss / len(dataloader)
        })
        # In a real async setup, we would await this.
        # Since this function is called from a synchronous background task wrapper,
        # we can't easily await. But we can use asyncio.run or run_coroutine_threadsafe if loop is available.
        # For simplicity, we will try to use the manager's broadcast in a fire-and-forget manner or wrapped.
        # Given FastAPI BackgroundTasks runs in a threadpool, we can't directly await.
        # We'll use a sync wrapper for broadcast if possible, or just skip for now in this context
        # but better to support it.
        try:
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             loop.run_until_complete(manager.broadcast(progress_msg))
             loop.close()
        except Exception as e:
             print(f"Failed to broadcast progress: {e}")


    # Evaluation
    model.eval()
    with torch.no_grad():
        if task_type == "classification":
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y_tensor).sum().item() / len(y_tensor)
            metrics = {"accuracy": acc}
        else:
            outputs = model(X_tensor)
            mse = criterion(outputs, y_tensor).item()
            metrics = {"mse": mse, "rmse": mse**0.5}

    # Save model state and metadata
    model_path = f"ml_models/model_{model_id}.pth"
    save_dict = {
        "state_dict": model.state_dict(),
        "metadata": {
            "input_dim": input_dim,
            "hidden_layers": hidden_layers,
            "output_dim": output_dim,
            "task_type": task_type
        }
    }
    torch.save(save_dict, model_path)
    return metrics, model_path

# --- Main Training Task ---

def train_model_task(model_id: int, dataset_id: int, model_info: dict):
    task_type = model_info.get('task_type')

    # Redirect to specialized training tasks
    if task_type == "image_classification":
        train_vision_model_task(model_id, dataset_id, model_info)
        return
    elif task_type == "text_generation":
        if model_type == "MoE":
            train_moe_model_task(model_id, dataset_id, model_info)
        else:
            train_mamba_model_task(model_id, dataset_id, model_info)
        return
    elif task_type == "alignment":
        train_dpo_task(model_id, dataset_id, model_info)
        return
    elif task_type == "data_generation":
        generate_synthetic_data_task(model_id, dataset_id, model_info)
        return
    elif task_type == "evaluation":
        llm_judge_task(model_id, dataset_id, model_info)
        return

    db: Session = SessionLocal()
    try:
        crud.update_model_status(db, model_id, "running")

        # Broadcast start
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "started"})))

        dataset = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset.filename}"

        # Basic check for tabular vs non-tabular
        if task_type not in ["image_classification", "text_generation"]:
             if dataset.dataset_type != "tabular":
                 raise ValueError(f"Task type '{task_type}' requires a tabular dataset, but got '{dataset.dataset_type}'.")
             df = pd.read_csv(file_location) if dataset.filename.endswith('.csv') else pd.read_excel(file_location)

        model_type = model_info.get('model_type')
        # task_type already retrieved
        hyperparams = model_info.get('hyperparameters', {})
        target_column = model_info.get('target_column')

        # Find the model class from the registry
        model_class = TASK_REGISTRY.get(task_type, {}).get(model_type)
        if not model_class:
            raise ValueError(f"Model type '{model_type}' not found for task '{task_type}'")

        # --- Data Preparation ---
        if task_type in ["classification", "regression"]:
            if not target_column:
                raise ValueError(f"Target column is required for {task_type}")

            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Preprocessing Pipeline
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y)
                # Save LabelEncoder for inference decoding (basic workaround)
                # In production, we'd pickle the label encoder or wrapper class

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        else:
            # For clustering/PCA
            X = df # Use all columns
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            X_scaled = preprocessor.fit_transform(X) # This might be sparse

        # --- Model Training & Evaluation ---
        metrics = {}
        if model_class == "pytorch": # Handling PyTorch case
            # NOTE: Pytorch part needs heavy refactoring to support mixed types properly.
            # For now, we keep the old logic for PyTorch but use only numeric columns to avoid breaking.
            # Ideally, we would use embeddings for categorical data.
            metrics, model_path = train_pytorch_model(df, target_column, hyperparams, model_id, task_type)

        elif model_class == "auto": # AutoML Case
            if task_type not in ["classification", "regression"]:
                raise ValueError("AutoML is currently only supported for classification and regression.")

            best_score = -float('inf')
            best_model = None
            best_pipeline = None
            best_metrics = None

            # Define candidate models
            if task_type == "classification":
                candidates = [
                    ("RandomForest", RandomForestClassifier(n_estimators=100)),
                    ("XGBoost", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                    ("DecisionTree", DecisionTreeClassifier())
                ]
                score_metric = "f1_score"
            else: # regression
                candidates = [
                    ("RandomForest", RandomForestRegressor(n_estimators=100)),
                    ("XGBoost", xgb.XGBRegressor()),
                    ("LinearRegression", LinearRegression())
                ]
                score_metric = "r2_score" # maximize r2

            for name, clf in candidates:
                 # Create pipeline
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('model', clf)])

                model_pipeline.fit(X_train, y_train)
                preds = model_pipeline.predict(X_test)

                if task_type == "classification":
                     score = f1_score(y_test, preds, average='weighted', zero_division=0)
                     current_metrics = {
                        "accuracy": accuracy_score(y_test, preds),
                        "precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                        "recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                        "f1_score": score,
                    }
                else:
                    score = r2_score(y_test, preds)
                    current_metrics = {
                        "mse": mean_squared_error(y_test, preds),
                        "rmse": mean_squared_error(y_test, preds, squared=False),
                        "mae": mean_absolute_error(y_test, preds),
                        "r2_score": score
                    }

                if score > best_score:
                    best_score = score
                    best_model = clf
                    best_pipeline = model_pipeline
                    best_metrics = current_metrics

            # Finalize best model
            model = best_pipeline
            metrics = best_metrics
            # Save the best model
            model_path = f"ml_models/model_{model_id}.joblib"
            joblib.dump(model, model_path)

        else:
            if task_type in ["classification", "regression"]:
                # Create a full pipeline
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('model', model_class(**hyperparams))])

                model_pipeline.fit(X_train, y_train)
                preds = model_pipeline.predict(X_test)

                # Save the full pipeline
                model = model_pipeline

                if task_type == "classification":
                    metrics = {
                        "accuracy": accuracy_score(y_test, preds),
                        "precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                        "recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                        "f1_score": f1_score(y_test, preds, average='weighted', zero_division=0),
                    }
                elif task_type == "regression":
                    metrics = {
                        "mse": mean_squared_error(y_test, preds),
                        "rmse": mean_squared_error(y_test, preds, squared=False),
                        "mae": mean_absolute_error(y_test, preds),
                        "r2_score": r2_score(y_test, preds)
                    }

            elif task_type == "clustering":
                # For clustering, we fit on preprocessed data
                model = model_class(**hyperparams)
                labels = model.fit_predict(X_scaled)

                # We can't save pipeline easily if we split fit_predict like this for metrics
                # So we will save a pipeline of [preprocessor, model]
                # But model is already fitted on X_scaled.

                if model_type == 'DBSCAN' and len(set(labels)) <= 2:
                     metrics = {"clusters_found": len(set(labels)) - (1 if -1 in labels else 0)}
                else:
                    # Silhouette score can be expensive on large data
                    if X_scaled.shape[0] < 10000:
                        try:
                            sil_score = silhouette_score(X_scaled, labels)
                        except:
                            sil_score = -1
                    else:
                        sil_score = -1 # Skip for speed

                    metrics = {
                        "silhouette_score": sil_score,
                        "clusters_found": len(set(labels))
                    }

                # Construct pipeline for saving (so inference works)
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                model = model_pipeline

            elif task_type == "dimensionality_reduction":
                model = model_class(**hyperparams)
                model.fit(X_scaled)
                metrics = {
                    "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                    "n_components": model.n_components_
                }
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                model = model_pipeline

            # Save the Scikit-learn/XGBoost model
            model_path = f"ml_models/model_{model_id}.joblib"
            joblib.dump(model, model_path)

        crud.update_model_status(db, model_id, "completed", metrics=metrics, model_path=model_path)

        # Broadcast completion
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed", "metrics": metrics})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
        print(f"Training failed for model_id {model_id}: {e}")

    finally:
        db.close()
