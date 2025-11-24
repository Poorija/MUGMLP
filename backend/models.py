from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_superuser = Column(Boolean, default=False)
    force_change_password = Column(Boolean, default=False)
    otp_secret = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    projects = relationship("Project", back_populates="owner")
    activity_logs = relationship("ActivityLog", back_populates="user")

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String, index=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="activity_logs")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_metadata = Column(Text) # Storing metadata as JSON string
    dataset_type = Column(String, default="tabular") # tabular, image_folder, text_file
    created_at = Column(DateTime, default=datetime.utcnow)
    project_id = Column(Integer, ForeignKey("projects.id"))

    project = relationship("Project", back_populates="datasets")
    ml_models = relationship("MLModel", back_populates="dataset")

class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    task_type = Column(String) # e.g., "classification"
    model_type = Column(String) # e.g., "KNeighborsClassifier"
    target_column = Column(String, nullable=True)
    status = Column(String, default="pending") # pending, running, completed, failed
    hyperparameters = Column(Text) # JSON string
    evaluation_metrics = Column(Text, nullable=True) # JSON string
    model_path = Column(String, nullable=True) # Path to the saved model file
    created_at = Column(DateTime, default=datetime.utcnow)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))

    dataset = relationship("Dataset", back_populates="ml_models")
