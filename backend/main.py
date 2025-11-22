from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import json
import os
from fastapi.responses import StreamingResponse

from . import models, schemas, crud, security
from .database import engine, get_db
from .dependencies import get_current_user, get_current_superuser
from .training import train_model_task, SimpleNN
from . import captcha_handler, visualizations
from .websocket_manager import manager
from . import activity
import joblib
import pyotp
import torch

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# Allow all origins for testing
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We keep the connection open.
            # The client might send messages if needed (e.g., subscribe to specific model updates)
            # For now, we just broadcast everything to everyone for simplicity.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Captcha Endpoint ---
@app.get("/captcha")
async def get_captcha():
    session_id, image_bytes = captcha_handler.generate_captcha()
    # Return session_id in a custom header
    return StreamingResponse(image_bytes, media_type="image/png", headers={'X-Captcha-Session-Id': session_id})


# --- Authentication Endpoints ---
@app.post("/token", response_model=schemas.Token)
def login_for_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
    # Optional captcha fields in headers or query params, but OAuth2PasswordRequestForm is standard.
    # We can check captcha separately if passed in headers, or we might need a custom login endpoint.
    # For now, let's assume specific headers for captcha if provided, or skip for basic auth flow?
    # The prompt asked to ADD captcha for login.
    captcha_session_id: str = Depends(lambda x: x.headers.get("X-Captcha-Session-Id")),
    captcha_text: str = Depends(lambda x: x.headers.get("X-Captcha-Text")),
    otp_code: str = Depends(lambda x: x.headers.get("X-OTP-Code"))
):
    # 1. Verify Captcha
    if not captcha_session_id or not captcha_text or not captcha_handler.verify_captcha(captcha_session_id, captcha_text):
        raise HTTPException(status_code=400, detail="Invalid or missing captcha")

    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Check 2FA
    if user.otp_secret:
        if not otp_code:
            # Tell client 2FA is required. But 401 is for auth failure.
            # We can return a 403 or custom error, or just fail.
            # Better: return 401 with specific detail so client knows to prompt 2FA
            # But here we just check if provided.
             raise HTTPException(status_code=401, detail="2FA Required")

        totp = pyotp.TOTP(user.otp_secret)
        if not totp.verify(otp_code):
            raise HTTPException(status_code=401, detail="Invalid 2FA Code")

    # 3. Check Force Change Password
    require_password_change = user.force_change_password

    # Log Activity
    activity.create_activity_log(db, user_id=user.id, action="login", details="User logged in")

    # Create token
    access_token = security.create_access_token(data={"sub": user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "require_password_change": require_password_change,
        "require_2fa": bool(user.otp_secret) # Just info for client, though verified above
    }

@app.get("/users/me", response_model=schemas.User)
def read_users_me(current_user: schemas.User = Depends(get_current_user)):
    return current_user

@app.put("/users/me", response_model=schemas.User)
def update_user_me(
    user_update: schemas.UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    if user_update.email:
        # Check if email is taken
        existing_user = crud.get_user_by_email(db, email=user_update.email)
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_update.email

    db.commit()
    db.refresh(current_user)
    activity.create_activity_log(db, user_id=current_user.id, action="update_profile", details="User updated profile details")
    return current_user

@app.get("/users/history", response_model=List[schemas.ActivityLog])
def read_user_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    return activity.get_activity_logs(db, user_id=current_user.id, skip=skip, limit=limit)

@app.post("/auth/change-password")
def change_password(
    password_update: schemas.UserPasswordUpdate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    # Verify old password
    if not security.verify_password(password_update.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")

    # Validate new password strength
    if not security.validate_password_strength(password_update.new_password):
        raise HTTPException(status_code=400, detail="Password does not meet strength requirements")

    # Update
    current_user.hashed_password = security.get_password_hash(password_update.new_password)
    current_user.force_change_password = False
    db.commit()
    activity.create_activity_log(db, user_id=current_user.id, action="change_password", details="User changed password")
    return {"message": "Password updated successfully"}

@app.post("/auth/setup-2fa")
def setup_2fa(current_user: schemas.User = Depends(get_current_user), db: Session = Depends(get_db)):
    secret = pyotp.random_base32()
    auth_url = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.email, issuer_name="ML Platform")
    return {"secret": secret, "otpauth_url": auth_url}

@app.post("/auth/enable-2fa")
def enable_2fa(
    payload: schemas.User2FASetup,
    code: str, # passed as query param or we can make it part of body
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    # This endpoint verifies the code against the provided secret (from frontend) and SAVES it to user
    totp = pyotp.TOTP(payload.otp_secret)
    if totp.verify(code):
        current_user.otp_secret = payload.otp_secret
        db.commit()
        activity.create_activity_log(db, user_id=current_user.id, action="enable_2fa", details="User enabled 2FA")
        return {"message": "2FA enabled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid code")

@app.post("/auth/disable-2fa")
def disable_2fa(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    current_user.otp_secret = None
    db.commit()
    activity.create_activity_log(db, user_id=current_user.id, action="disable_2fa", details="User disabled 2FA")
    return {"message": "2FA disabled successfully"}

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Verify captcha first
    if not captcha_handler.verify_captcha(user.captcha_session_id, user.captcha_text):
        raise HTTPException(status_code=400, detail="Invalid captcha")

    # Verify Password Strength
    if not security.validate_password_strength(user.password):
        raise HTTPException(status_code=400, detail="Password too weak. Minimum 8 chars, mixed letters/numbers.")

    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

# --- Project Endpoints ---
@app.post("/projects/", response_model=schemas.Project)
def create_project(
    project: schemas.ProjectCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    return crud.create_project(db=db, project=project, owner_id=current_user.id)

@app.get("/projects/", response_model=List[schemas.Project])
def read_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    projects = crud.get_projects_by_owner(db, owner_id=current_user.id, skip=skip, limit=limit)
    return projects

@app.get("/projects/{project_id}/datasets", response_model=List[schemas.Dataset])
def read_project_datasets(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    project = crud.get_project(db, project_id=project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")

    datasets = crud.get_datasets_by_project(db, project_id=project_id)
    return datasets

# --- Dataset Endpoints ---
@app.post("/projects/{project_id}/datasets/", response_model=schemas.Dataset)
async def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    project = crud.get_project(db, project_id=project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")

    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_location)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_location)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        metadata = {
            "columns": list(df.columns),
            "rows": len(df),
            "file_size": os.path.getsize(file_location),
        }
        metadata_str = json.dumps(metadata)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

    dataset = schemas.DatasetCreate(filename=file.filename, file_metadata=metadata_str)
    return crud.create_dataset(db=db, dataset=dataset, project_id=project_id)

def get_and_verify_dataset(dataset_id: int, db: Session, current_user: schemas.User) -> models.Dataset:
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = crud.get_project(db, project_id=dataset.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    return dataset

@app.get("/datasets/{dataset_id}")
def read_dataset_content(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    dataset = get_and_verify_dataset(dataset_id, db, current_user)
    file_location = f"uploads/{dataset.filename}"

    try:
        if dataset.filename.endswith('.csv'):
            df = pd.read_csv(file_location)
        else:
            df = pd.read_excel(file_location)

        # Return first 50 rows as JSON
        return df.head(50).to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found on server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

@app.get("/datasets/{dataset_id}/summary")
def get_dataset_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    dataset = get_and_verify_dataset(dataset_id, db, current_user)
    file_location = f"uploads/{dataset.filename}"

    try:
        if dataset.filename.endswith('.csv'):
            df = pd.read_csv(file_location)
        else:
            df = pd.read_excel(file_location)

        # Generate descriptive statistics
        # Convert to json split format, but then parse it so we return a dict
        summary = df.describe(include='all').to_json(orient="split")
        summary_dict = json.loads(summary)

        # Add data distribution info for histograms (simplified: only numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        distributions = {}
        for col in numeric_cols:
            # Create a simple histogram using numpy or just value_counts for discrete
            # For simplicity, we'll just send value counts for small unique sets or binning
            # Let's use pandas cut for binning
            try:
                counts = pd.cut(df[col], bins=10).value_counts(sort=False)
                distributions[col] = [{"bin": str(k), "count": int(v)} for k, v in counts.items()]
            except:
                pass

        return {"summary": summary_dict, "distributions": distributions}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found on server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# --- ML Model Training Endpoint ---
@app.post("/datasets/{dataset_id}/train", response_model=schemas.MLModel)
def train_model(
    dataset_id: int,
    model_create: schemas.MLModelCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    dataset = get_and_verify_dataset(dataset_id, db, current_user)

    # Create an initial record for the ML model in the database
    ml_model = crud.create_ml_model(db=db, model=model_create, dataset_id=dataset.id)

    # Add the training job to background tasks
    background_tasks.add_task(
        train_model_task,
        model_id=ml_model.id,
        dataset_id=dataset.id,
        model_info=model_create.dict()
    )

    return ml_model

@app.get("/datasets/{dataset_id}/models", response_model=List[schemas.MLModel])
def read_dataset_models(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    dataset = get_and_verify_dataset(dataset_id, db, current_user)
    return crud.get_ml_models_by_dataset(db, dataset_id=dataset.id)

@app.get("/models/{model_id}", response_model=schemas.MLModel)
def read_ml_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    model = crud.get_ml_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Verify user has access to the dataset this model belongs to
    get_and_verify_dataset(model.dataset_id, db, current_user)
    return model

# --- Admin Endpoints ---
@app.get("/admin/users", response_model=List[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_superuser)
):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

# --- Visualization Endpoint ---
@app.get("/models/{model_id}/visualizations/{vis_type}")
async def get_visualization(
    model_id: int,
    vis_type: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    ml_model = crud.get_ml_model(db, model_id)
    if not ml_model or not ml_model.model_path:
        raise HTTPException(status_code=404, detail="Model or model file not found")

    # Verify access
    dataset = get_and_verify_dataset(ml_model.dataset_id, db, current_user)

    try:
        model = joblib.load(ml_model.model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model file: {e}")

    dataset_path = f"uploads/{dataset.filename}"

    if vis_type == "confusion_matrix":
        if ml_model.task_type != "classification":
            raise HTTPException(status_code=400, detail="Confusion matrix is only for classification models")
        image_buf = visualizations.create_confusion_matrix(model, dataset_path, ml_model.target_column)
        return StreamingResponse(image_buf, media_type="image/png")

    elif vis_type == "cluster_scatter":
        if ml_model.task_type != "clustering":
            raise HTTPException(status_code=400, detail="Cluster plot is only for clustering models")
        image_buf = visualizations.create_cluster_scatter_plot(model, dataset_path)
        return StreamingResponse(image_buf, media_type="image/png")

    elif vis_type == "feature_importance":
        if ml_model.task_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="Feature importance is only for supervised models")

        # Check if model supports it (trees)
        estimator = model.named_steps['model'] if hasattr(model, 'named_steps') else model
        if not hasattr(estimator, 'feature_importances_'):
             raise HTTPException(status_code=400, detail="This model does not support feature importance")

        image_buf = visualizations.create_feature_importance_plot(model, dataset_path, ml_model.target_column)
        return StreamingResponse(image_buf, media_type="image/png")

    elif vis_type == "actual_vs_predicted":
        if ml_model.task_type != "regression":
             raise HTTPException(status_code=400, detail="Actual vs Predicted is only for regression models")
        image_buf = visualizations.create_actual_vs_predicted_plot(model, dataset_path, ml_model.target_column)
        return StreamingResponse(image_buf, media_type="image/png")

    else:
        raise HTTPException(status_code=404, detail="Visualization type not found")

# --- Prediction Endpoint ---
@app.post("/models/{model_id}/predict")
async def predict(
    model_id: int,
    data: dict, # Accepts JSON with keys matching feature names
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    ml_model = crud.get_ml_model(db, model_id)
    if not ml_model or not ml_model.model_path:
        raise HTTPException(status_code=404, detail="Model not found or training not complete")

    get_and_verify_dataset(ml_model.dataset_id, db, current_user)

    try:
        # Create DataFrame from input data
        # Expecting input like {"feature1": val1, "feature2": val2}
        # or {"features": [{"f1": v1}, {"f1": v2}]} for batch
        if "features" in data and isinstance(data["features"], list):
            input_df = pd.DataFrame(data["features"])
        else:
             input_df = pd.DataFrame([data])

        # Attempt to convert columns to numeric where possible, as API input is likely strings
        for col in input_df.columns:
            # We can try to convert to numeric, if it fails (e.g. categorical), it keeps as object/string
            input_df[col] = pd.to_numeric(input_df[col], errors='ignore')

        if ml_model.model_type == "SimpleNN":
            # PyTorch loading logic
            saved_data = torch.load(ml_model.model_path)
            metadata = saved_data["metadata"]

            model = SimpleNN(
                input_dim=metadata["input_dim"],
                hidden_layers=metadata["hidden_layers"],
                output_dim=metadata["output_dim"],
                task_type=metadata["task_type"]
            )
            model.load_state_dict(saved_data["state_dict"])
            model.eval()

            # For PyTorch SimpleNN, we assumed only numeric features in training.py
            # So we select only numeric columns.
            # Warning: This assumes input_df has same columns as training data.
            # A robust solution requires saving column names/transformers.
            # For now, we convert all to float32 tensor.
            X_tensor = torch.tensor(input_df.values, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(X_tensor)
                if metadata["task_type"] == "classification":
                    _, predictions = torch.max(outputs.data, 1)
                else:
                    predictions = outputs

            predictions = predictions.tolist()

        else:
            model = joblib.load(ml_model.model_path)
            # The pipeline handles preprocessing!
            predictions = model.predict(input_df)

            # Convert numpy/tensor to list
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            elif isinstance(predictions, pd.Series):
                predictions = predictions.tolist()

        return {"predictions": predictions}

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML App API"}
