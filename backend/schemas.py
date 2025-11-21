from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    captcha_session_id: str
    captcha_text: str

class UserPasswordUpdate(BaseModel):
    old_password: str
    new_password: str

class User(UserBase):
    id: int
    is_active: bool = True # Assuming we might add this later

    class Config:
        orm_mode = True

# --- MLModel Schemas ---
class MLModelBase(BaseModel):
    name: str
    model_type: str
    hyperparameters: dict

class MLModelCreate(MLModelBase):
    task_type: str  # e.g., "classification", "clustering"
    target_column: Optional[str] = None

class MLModel(MLModelBase):
    id: int
    status: str
    evaluation_metrics: Optional[dict] = None
    created_at: datetime
    dataset_id: int

    class Config:
        orm_mode = True

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str
    require_password_change: Optional[bool] = False
    require_2fa: Optional[bool] = False

class TokenData(BaseModel):
    email: Optional[str] = None

# --- Dataset Schemas ---
class DatasetBase(BaseModel):
    filename: str
    file_metadata: str # JSON string for simplicity

class DatasetCreate(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int
    created_at: datetime
    project_id: int

    class Config:
        orm_mode = True

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    owner_id: int
    created_at: datetime
    datasets: List[Dataset] = []

    class Config:
        orm_mode = True
