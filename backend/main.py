from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import json
import os
from fastapi.responses import StreamingResponse

from . import models, schemas, crud, security
from .database import engine, get_db
from .dependencies import get_current_user
from .training import train_model_task
from . import captcha_handler, visualizations
import joblib

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# --- Captcha Endpoint ---
@app.get("/captcha")
async def get_captcha():
    session_id, image_bytes = captcha_handler.generate_captcha()
    # Return session_id in a custom header
    return StreamingResponse(image_bytes, media_type="image/png", headers={'X-Captcha-Session-Id': session_id})


# --- Authentication Endpoints ---
@app.post("/token", response_model=schemas.Token)
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Verify captcha first
    if not captcha_handler.verify_captcha(user.captcha_session_id, user.captcha_text):
        raise HTTPException(status_code=400, detail="Invalid captcha")

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

    dataset = schemas.DatasetCreate(filename=file.filename, metadata=metadata_str)
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
        summary = df.describe(include='all').to_json(orient="split")
        return json.loads(summary)
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

    else:
        raise HTTPException(status_code=404, detail="Visualization type not found")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML App API"}
