from sqlalchemy.orm import Session
from . import models, schemas
from .security import get_password_hash
from typing import List

# --- User CRUD ---
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Project CRUD ---
def get_projects_by_owner(db: Session, owner_id: int, skip: int = 0, limit: int = 100) -> List[models.Project]:
    return db.query(models.Project).filter(models.Project.owner_id == owner_id).offset(skip).limit(limit).all()

def create_project(db: Session, project: schemas.ProjectCreate, owner_id: int):
    db_project = models.Project(**project.dict(), owner_id=owner_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_project(db: Session, project_id: int):
    return db.query(models.Project).filter(models.Project.id == project_id).first()

# --- Dataset CRUD ---
def get_dataset(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()

def get_datasets_by_project(db: Session, project_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Dataset).filter(models.Dataset.project_id == project_id).offset(skip).limit(limit).all()

def create_dataset(db: Session, dataset: schemas.DatasetCreate, project_id: int):
    db_dataset = models.Dataset(**dataset.dict(), project_id=project_id)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

# --- MLModel CRUD ---
def create_ml_model(db: Session, model: schemas.MLModelCreate, dataset_id: int):
    import json
    db_model = models.MLModel(
        name=model.name,
        task_type=model.task_type,
        model_type=model.model_type,
        target_column=model.target_column,
        hyperparameters=json.dumps(model.hyperparameters),
        dataset_id=dataset_id,
        status="pending"
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def update_model_status(db: Session, model_id: int, status: str, metrics: dict = None, model_path: str = None):
    import json
    db_model = db.query(models.MLModel).filter(models.MLModel.id == model_id).first()
    if db_model:
        db_model.status = status
        if metrics:
            db_model.evaluation_metrics = json.dumps(metrics)
        if model_path:
            db_model.model_path = model_path
        db.commit()
        db.refresh(db_model)
    return db_model

def get_ml_model(db: Session, model_id: int):
    return db.query(models.MLModel).filter(models.MLModel.id == model_id).first()

def get_ml_models_by_dataset(db: Session, dataset_id: int):
    return db.query(models.MLModel).filter(models.MLModel.dataset_id == dataset_id).all()
