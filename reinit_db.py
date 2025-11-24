from backend.database import engine, Base
from backend.models import User, Project, Dataset, MLModel, ActivityLog

def reinit_db():
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Database re-initialized successfully.")

if __name__ == "__main__":
    reinit_db()
