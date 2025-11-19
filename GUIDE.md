# User Guide: Your Journey with the ML Platform

Welcome to the ML Platform! This guide will walk you through the key features of the application, from creating your account to training your first machine learning model.

## 1. 📝 Creating Your Account

Before you can start, you need to create an account.

-   Navigate to the **Register** page.
-   Fill in your email and a secure password.
-   **CAPTCHA**: For security, you'll need to solve a CAPTCHA. Type the characters you see in the image into the text box. If you can't read it, click the "Refresh Captcha" button to get a new one.
-   Click **Register**. If successful, you'll be redirected to the login page.

## 2. 🔐 Logging In

-   Navigate to the **Login** page.
-   Enter the email and password you registered with.
-   You will be redirected to your **Dashboard** upon successful login.

## 3. 🗂️ Managing Projects (Your Workspace)

The **Dashboard** is your central hub. All your work is organized into **Projects**.

-   **Create a Project**: Give your new project a name in the "Create New Project" form and click **Create**. It will appear in the "Your Projects" list.
-   **View a Project**: Click on any project name in the list to go to its **Project Details** page.

## 4. 📊 Uploading and Exploring Datasets

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

## 5. 🤖 Training a Machine Learning Model

This is where the magic happens!

1.  **Select a Dataset**: First, click on the dataset you want to use for training. This will load its data and open the training panel.

2.  **Configure the Training Job**:
    -   **Model Name**: Give your trained model a descriptive name (e.g., "Iris_KNN_Model").
    -   **Target Column**: From the dropdown, select the column that you want the model to predict. This is your `y` variable.
    -   **Model Type**: Choose the algorithm you want to use:
        -   **Classical Models**: `KNN`, `Decision Tree`, `Random Forest`.
        -   **Deep Learning**: `Simple Neural Network`.

3.  **Set Hyperparameters**:
    -   If you choose a classical model, you can set its specific parameters (this feature will be expanded).
    -   If you choose `SimpleNN`, you must define the neural network's architecture:
        -   **Hidden Layers**: Define the number of neurons in each hidden layer, separated by commas (e.g., `64,32` for two hidden layers with 64 and 32 neurons).
        -   **Epochs**: The number of times the model will see the entire dataset during training.
        -   **Learning Rate**: A small number that controls the step size during model optimization.

4.  **Start Training**:
    -   Click the **Train Model** button.
    -   The training process will start in the background. You don't need to wait on the page.

## 6. 📈 Viewing Results (Coming Soon!)

Once the training is complete, a new section will appear on the **Project Details** page:

-   A list of all models trained on the selected dataset.
-   The **status** of each model (`pending`, `running`, `completed`, `failed`).
-   For completed models, you'll be able to click to see:
    -   **Evaluation Metrics**: How well the model performed (e.g., Accuracy, Precision).
    -   **Visualizations**: Charts and plots, like a Confusion Matrix, to help you understand the model's results.

Happy modeling!
