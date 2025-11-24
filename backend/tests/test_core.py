import pytest
from backend import security
from backend.main import app

def test_password_hashing():
    password = "secret_password"
    hashed = security.get_password_hash(password)
    assert security.verify_password(password, hashed)
    assert not security.verify_password("wrong_password", hashed)

def test_password_strength():
    assert security.validate_password_strength("password123") is True
    assert security.validate_password_strength("Pass1234") is True
    assert security.validate_password_strength("short1") is False # Too short
    assert security.validate_password_strength("onlyletters") is False # No number
    assert security.validate_password_strength("12345678") is False # No letter

def test_create_user(client, mocker):
    # Mock captcha verification
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)

    response = client.post(
        "/users/",
        json={
            "email": "test@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_create_user_existing_email(client, mocker):
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)

    # Create first user
    client.post(
        "/users/",
        json={
            "email": "duplicate@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )

    # Try creating again
    response = client.post(
        "/users/",
        json={
            "email": "duplicate@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_login_success(client, mocker):
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)

    # Create user
    client.post(
        "/users/",
        json={
            "email": "login@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )

    # Login
    response = client.post(
        "/token",
        data={"username": "login@example.com", "password": "StrongPassword1"},
        headers={"X-Captcha-Session-Id": "dummy", "X-Captcha-Text": "dummy"}
    )
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_wrong_password(client, mocker):
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)

    # Create user
    client.post(
        "/users/",
        json={
            "email": "wrongpass@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )

    # Login
    response = client.post(
        "/token",
        data={"username": "wrongpass@example.com", "password": "WrongPassword"},
        headers={"X-Captcha-Session-Id": "dummy", "X-Captcha-Text": "dummy"}
    )
    assert response.status_code == 401

def test_read_users_me(client, mocker):
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)

    # Create user
    client.post(
        "/users/",
        json={
            "email": "me@example.com",
            "password": "StrongPassword1",
            "captcha_session_id": "dummy",
            "captcha_text": "dummy"
        }
    )

    # Login
    login_res = client.post(
        "/token",
        data={"username": "me@example.com", "password": "StrongPassword1"},
        headers={"X-Captcha-Session-Id": "dummy", "X-Captcha-Text": "dummy"}
    )
    token = login_res.json()["access_token"]

    # Read Me
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
