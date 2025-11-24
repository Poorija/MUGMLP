import pytest
import io
from backend.models import Dataset, Project

def test_full_project_flow(client, mocker):
    # 1. Login
    mocker.patch("backend.captcha_handler.verify_captcha", return_value=True)
    client.post("/users/", json={"email": "flow@example.com", "password": "StrongPassword1", "captcha_session_id": "d", "captcha_text": "d"})
    login = client.post("/token", data={"username": "flow@example.com", "password": "StrongPassword1"}, headers={"X-Captcha-Session-Id": "d", "X-Captcha-Text": "d"})
    token = login.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    # 2. Create Project
    res = client.post("/projects/", json={"name": "Flow Project"}, headers=auth)
    assert res.status_code == 200
    project_id = res.json()["id"]

    # 3. Upload Dataset
    csv_content = b"col1,col2,target\n1,2,0\n3,4,1"
    files = {"file": ("data.csv", io.BytesIO(csv_content), "text/csv")}
    res = client.post(f"/projects/{project_id}/datasets/", files=files, headers=auth)
    assert res.status_code == 200
    dataset_id = res.json()["id"]

    # 4. List Datasets
    res = client.get(f"/projects/{project_id}/datasets", headers=auth)
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["id"] == dataset_id

    # 5. Get Dataset Content
    res = client.get(f"/datasets/{dataset_id}", headers=auth)
    assert res.status_code == 200
    assert len(res.json()) == 2 # 2 rows

    # 6. Get Dataset Summary
    res = client.get(f"/datasets/{dataset_id}/summary", headers=auth)
    assert res.status_code == 200
    assert "summary" in res.json()
    assert "distributions" in res.json()
