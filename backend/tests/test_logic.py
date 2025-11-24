import pytest
from unittest.mock import MagicMock
from backend import training, rag
import pandas as pd
import io

# Mock the rag engine
@pytest.fixture
def mock_rag_engine(mocker):
    mock = mocker.patch("backend.main.rag_engine")
    mock.ingest_document.return_value = 5  # 5 chunks
    mock.query.return_value = {"answer": "Test answer", "sources": []}
    return mock

# Mock the training task
@pytest.fixture
def mock_training_task(mocker):
    return mocker.patch("backend.main.train_model_task")

def test_rag_ingest(client, mock_rag_engine, mocker):
    # Create user and login
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)
    client.post("/users/", json={"email": "rag@example.com", "password": "StrongPassword1", "captcha_session_id": "d", "captcha_text": "d"})
    login = client.post("/token", data={"username": "rag@example.com", "password": "StrongPassword1"}, headers={"X-Captcha-Session-Id": "d", "X-Captcha-Text": "d"})
    token = login.json()["access_token"]

    # Upload file
    files = {"file": ("test.txt", io.BytesIO(b"This is a test document."), "text/plain")}
    response = client.post(
        "/rag/ingest",
        files=files,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["chunks"] == 5
    mock_rag_engine.ingest_document.assert_called_once()

def test_rag_query(client, mock_rag_engine, mocker):
    # Auth
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)
    client.post("/users/", json={"email": "rag2@example.com", "password": "StrongPassword1", "captcha_session_id": "d", "captcha_text": "d"})
    login = client.post("/token", data={"username": "rag2@example.com", "password": "StrongPassword1"}, headers={"X-Captcha-Session-Id": "d", "X-Captcha-Text": "d"})
    token = login.json()["access_token"]

    response = client.post(
        "/rag/query",
        json={"text": "What is this?"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"

def test_train_model_trigger(client, mock_training_task, mocker):
    # Auth
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)
    client.post("/users/", json={"email": "train@example.com", "password": "StrongPassword1", "captcha_session_id": "d", "captcha_text": "d"})
    login = client.post("/token", data={"username": "train@example.com", "password": "StrongPassword1"}, headers={"X-Captcha-Session-Id": "d", "X-Captcha-Text": "d"})
    token = login.json()["access_token"]

    # Create Project & Dataset first
    client.post("/projects/", json={"name": "TrainProj"}, headers={"Authorization": f"Bearer {token}"})
    files = {"file": ("data.csv", io.BytesIO(b"col1,col2,target\n1,2,0\n3,4,1"), "text/csv")}
    ds_res = client.post("/projects/1/datasets/", files=files, headers={"Authorization": f"Bearer {token}"})
    dataset_id = ds_res.json()["id"]

    # Trigger Training
    response = client.post(
        f"/datasets/{dataset_id}/train",
        json={
            "name": "MyModel",
            "task_type": "classification",
            "model_type": "LogisticRegression",
            "target_column": "target",
            "hyperparameters": {}
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    # Verify background task was added
    mock_training_task.assert_called_once()

def test_predict_endpoint_simplenn(client, mocker):
    # Mock loading a SimpleNN model
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)
    client.post("/users/", json={"email": "pred@example.com", "password": "StrongPassword1", "captcha_session_id": "d", "captcha_text": "d"})
    login = client.post("/token", data={"username": "pred@example.com", "password": "StrongPassword1"}, headers={"X-Captcha-Session-Id": "d", "X-Captcha-Text": "d"})
    token = login.json()["access_token"]

    # Setup mock model in DB
    # We need to manually insert a model record or go through the flow.
    # Going through flow is cleaner but requires mocking the training completion.
    # Let's mock crud.get_ml_model and the model loading.

    mock_model_db = MagicMock()
    mock_model_db.model_path = "dummy_path.pt"
    mock_model_db.model_type = "SimpleNN"
    mock_model_db.task_type = "classification"
    mock_model_db.dataset_id = 1

    mocker.patch("backend.crud.get_ml_model", return_value=mock_model_db)
    mocker.patch("backend.main.get_and_verify_dataset") # Skip dataset verification logic

    # Mock torch load
    mock_torch_load = mocker.patch("torch.load")
    mock_torch_load.return_value = {
        "metadata": {"input_dim": 2, "hidden_layers": [10], "output_dim": 2, "task_type": "classification"},
        "state_dict": MagicMock()
    }

    # Mock SimpleNN class and its forward pass
    MockSimpleNN = mocker.patch("backend.main.SimpleNN")
    mock_nn_instance = MockSimpleNN.return_value
    mock_nn_instance.return_value = mocker.Mock(data=mocker.Mock()) # outputs
    # Mock torch.max for classification
    mocker.patch("torch.max", return_value=(None, mocker.Mock(tolist=lambda: [1])))
    mocker.patch("torch.tensor")

    response = client.post(
        "/models/1/predict",
        json={"features": [{"col1": 1, "col2": 2}]},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["predictions"] == [1]
