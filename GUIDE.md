# User Guide: Your Journey with the ML Platform (2025 Edition)

Welcome to the ML Platform! This guide will walk you through the key features of the application, now upgraded with a modern UI and enhanced security features.

## 1. 📝 Creating Your Account

Before you can start, you need to create an account.

-   Navigate to the **Register** page.
-   Fill in your email and a secure password.
-   **CAPTCHA**: For security, you'll need to solve a CAPTCHA. Type the characters you see in the image into the text box. If you can't read it, click the "Refresh Captcha" button to get a new one.
-   Click **Register**. If successful, you'll be redirected to the login page.

## 2. 🔐 Logging In & Security

-   Navigate to the **Login** page.
-   Enter the email and password you registered with.
-   **2FA (Two-Factor Authentication)**: If you have enabled 2FA on your profile, you will be prompted to enter the code from your authenticator app (like Google Authenticator).

## 3. 👤 User Profile & Settings (New!)

Access your profile by clicking "Profile" in the navigation menu.

-   **Profile Info**: Update your email address.
-   **Security**:
    -   **Change Password**: Update your password securely.
    -   **Two-Factor Authentication**: Scan the QR code with an app like Google Authenticator to enable an extra layer of security.
-   **Activity History**: View a log of your recent actions (logins, uploads, training jobs) to monitor your account usage.

## 4. 🗂️ Managing Projects (Your Workspace)

The **Dashboard** is your central hub. All your work is organized into **Projects**.

-   **Create a Project**: Give your new project a name in the "Create New Project" form and click **Create**. It will appear in the "Your Projects" list.
-   **View a Project**: Click on any project name in the list to go to its **Project Details** page.

## 5. 📊 Uploading and Exploring Datasets

Inside a project, you can manage your datasets.

-   **Upload a Dataset**:
    -   On the **Project Details** page, you'll find the "Upload New Dataset" form.
    -   Click the file input to select a **CSV** or **Excel** file from your computer.
    -   Click **Upload**. The dataset will appear in the "Datasets" list for that project.
-   **Explore a Dataset**:
    -   Click on a dataset's name from the list.
    -   The platform will display:
        -   A **Data Preview**: The first 50 rows of your dataset in a table.
        -   A **Statistical Summary**: Key statistics for each column (like mean, standard deviation, count, etc.).

## 6. 🤖 Training a Machine Learning Model

This is where the magic happens!

1.  **Select a Dataset**: First, click on the dataset you want to use for training. This will load its data and open the training panel.

2.  **Configure the Training Job**:
    -   **Model Name**: Give your trained model a descriptive name (e.g., "Iris_KNN_Model").
    -   **Target Column**: From the dropdown, select the column that you want the model to predict. This is your `y` variable.
    -   **Model Type**: Choose the algorithm you want to use:
        -   **Classical Models**: `KNN`, `Decision Tree`, `Random Forest`.
        -   **Deep Learning**: `Simple Neural Network`.

3.  **Set Hyperparameters**:
    -   If you choose a classical model, you can set its specific parameters.
    -   If you choose `SimpleNN`, you must define the neural network's architecture.

4.  **Start Training**:
    -   Click the **Train Model** button.
    -   The training process will start in the background.

## 7. 💡 Global Tooltips

As you navigate the application, hover over buttons, fields, and icons to see quick tips and explanations of what each feature does. This helps you learn the platform without needing to constantly check the documentation.

Happy modeling!
