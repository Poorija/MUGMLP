import asyncio
from backend.database import SessionLocal
from backend.models import User
from backend.security import get_password_hash
import os

async def main():
    db = SessionLocal()
    try:
        # Check if an admin user already exists
        if db.query(User).filter(User.is_superuser == True).first():
            print("Admin user already exists.")
            return

        # Create a new admin user
        email = os.environ.get("ADMIN_EMAIL", "admin") # User: admin
        password = os.environ.get("ADMIN_PASSWORD", "P@ssw0rd")

        admin_user = User(
            email=email,
            hashed_password=get_password_hash(password),
            is_superuser=True,
            force_change_password=True
        )
        db.add(admin_user)
        db.commit()
        print(f"Admin user {email} created successfully.")

    finally:
        db.close()

if __name__ == "__main__":
    # To run this script, execute: python -m create_admin
    # You might need to adjust PYTHONPATH if imports fail
    # For example: PYTHONPATH=. python create_admin.py
    asyncio.run(main())
