from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect, Header
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import json
import os
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import models, schemas, crud, security
from database import engine, get_db
from dependencies import get_current_user, get_current_superuser
from training import train_model_task, SimpleNN
import captcha_handler, visualizations
from websocket_manager import manager
import activity
from hardware_scanner import scanner
from rag import rag_engine
import joblib
import pyotp
import torch

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# فعال‌سازی CORS برای ارتباط درست frontend با backend و ارسال هدر کپچا
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Captcha-Session-Id"],
)

# سایر کدها مثل قبل باقی می‌ماند ...
# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Captcha Endpoint ---
@app.get("/captcha")
async def get_captcha():
    session_id, image_bytes = captcha_handler.generate_captcha()
    return StreamingResponse(image_bytes, media_type="image/png", headers={'X-Captcha-Session-Id': session_id})

# --- Authentication Endpoints ---
@app.post("/token", response_model=schemas.Token)
def login_for_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
    captcha_session_id: Optional[str] = Header(None, alias="X-Captcha-Session-Id"),
    captcha_text: Optional[str] = Header(None, alias="X-Captcha-Text"),
    otp_code: Optional[str] = Header(None, alias="X-OTP-Code")
):
    if not captcha_session_id or not captcha_text or not captcha_handler.verify_captcha(captcha_session_id, captcha_text):
        raise HTTPException(status_code=400, detail="Invalid or missing captcha")

    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.otp_secret:
        if not otp_code:
             raise HTTPException(status_code=401, detail="2FA Required")
        totp = pyotp.TOTP(user.otp_secret)
        if not totp.verify(otp_code):
            raise HTTPException(status_code=401, detail="Invalid 2FA Code")

    require_password_change = user.force_change_password

    activity.create_activity_log(db, user_id=user.id, action="login", details="User logged in")

    access_token = security.create_access_token(data={"sub": user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "require_password_change": require_password_change,
        "require_2fa": bool(user.otp_secret)
    }

# سایر endpointها مثل قبل باقی‌می‌ماند ...
