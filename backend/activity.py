from sqlalchemy.orm import Session
from . import models, schemas
import json

def create_activity_log(db: Session, user_id: int, action: str, details: str = None):
    """
    Logs a user activity.
    """
    db_log = models.ActivityLog(user_id=user_id, action=action, details=details)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_activity_logs(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    """
    Retrieves activity logs for a specific user.
    """
    return db.query(models.ActivityLog).filter(models.ActivityLog.user_id == user_id).order_by(models.ActivityLog.timestamp.desc()).offset(skip).limit(limit).all()
