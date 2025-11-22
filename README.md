# Comprehensive ML Platform

This project is a web-based, end-to-end platform for machine learning tasks, designed to take users from data upload and preprocessing to model training and evaluation. It leverages a modern tech stack to provide a responsive and powerful user experience, featuring a cutting-edge **2025-ready UI/UX**.

## ✨ Features

- **User Authentication**: Secure JWT-based registration and login system with CAPTCHA protection and **Two-Factor Authentication (2FA)**.
- **User Profile**: Manage account settings, change passwords, and view **Activity History**.
- **Modern UI/UX**:
    - **Glassmorphism Design**: Translucent elements and depth.
    - **Smooth Animations**: Powered by `framer-motion` for seamless transitions.
    - **Global Tooltips**: Context-aware help system across the entire app.
- **Project & Dataset Management**: Users can create projects to organize their work and upload datasets (CSV, Excel).
- **Data Exploration**: Preview uploaded data and view detailed statistical summaries.
- **Asynchronous Model Training**:
  - **Classical ML**: Train models like KNN, Decision Trees, and Random Forests using Scikit-learn.
  - **Deep Learning**: Train custom neural networks using PyTorch.
- **Admin Panel**: Backend infrastructure for an admin user to manage the platform.
- **Containerized**: Fully containerized with Docker for easy setup and deployment.

## 🚀 Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React (JavaScript), Material UI, Framer Motion
- **Database**: PostgreSQL
- **Containerization**: Docker & Docker Compose
- **ML Libraries**: Scikit-learn, PyTorch, Pandas
- **Authentication**: JWT (JSON Web Tokens), PyOTP (2FA)
- **Security**: Locally-generated CAPTCHA

## 🛠️ Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose installed on your machine.
- For Windows/Mac users: **Docker Desktop**.

### Installation & Running the App

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Build and run the containers:**

    *For Linux:*
    ```bash
    sudo docker compose up --build -d
    ```

    *For Windows/Mac (Docker Desktop):*
    Open your terminal (PowerShell, Command Prompt, or Terminal) and run:
    ```bash
    docker-compose up --build -d
    ```
    *(Note: Ensure Docker Desktop is running before executing this command.)*

3.  **Create a superuser (Admin):**
    Run the `create_admin.py` script to create an initial admin account. You can set the admin credentials via environment variables (`ADMIN_EMAIL`, `ADMIN_PASSWORD`).

    *Linux:*
    ```bash
    sudo docker compose exec backend python create_admin.py
    ```
    *Windows/Mac (Docker Desktop):*
    ```bash
    docker-compose exec backend python create_admin.py
    ```

4.  **Access the application:**
    - **Frontend (React App)**: [http://localhost:3000](http://localhost:3000)
    - **Backend API (FastAPI)**: [http://localhost:8000](http://localhost:8000)
    - **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📁 Project Structure

```
.
├── backend/         # FastAPI application, all Python code
├── frontend/        # React application, all JS/CSS code
├── ml_models/       # Directory for storing saved trained models
├── uploads/         # Directory for storing uploaded datasets
├── create_admin.py  # Script to create an initial superuser
├── docker-compose.yml # Defines and configures all services
└── README.md        # This file
```
