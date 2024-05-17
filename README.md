# Flask API for Nu Prediction using ML and DL Models

This project is developed as part of a BTech project at IIT Roorkee. It involves using Machine Learning (ML) and Deep Learning (DL) models to predict Nu. The Flask API serves as the interface for users to interact with the prediction models.

## Author
**M Hitesh**
BTech, IIT Roorkee

## Project Overview
This project sets up a Flask API for predicting Nu using pre-trained ML and DL models. The API allows users to input data and receive predictions via HTTP requests.

## Setup Instructions

### Step 1: Download the Project Files
First, download the project files to your local machine. You can clone the repository into your desired folder (you need to have git installed in your machine or you can just download the zip folder and unzip it into your desired folder):
```bash
git clone https://github.com/Holmes629/Nu_Prediction_Models.git
```

### Step 2: Install the Required Packages
Navigate to the project directory where requirements.txt is located and install the necessary packages using pip:
```bash
cd path/to/project/directory
pip install -r requirements.txt
```

### Step 3: Run the Flask Application
After installing the required packages, run the Flask application using the following command:
```bash
python index.py
```

### Step 4: Access the API
Open your web browser and go to:
```bash
http://127.0.0.1:5000/
```

Here, you can interact with the API to make predictions.
API Endpoints

The primary endpoint for prediction is:

    /predict: This endpoint accepts input data and returns the predicted Nu value.

Example Request

To make a prediction, send a POST request to /predict with the required input data.
Dependencies

The project relies on several Python packages which are listed in requirements.txt. Ensure that you have Python installed on your system before installing these dependencies.

## License
This project is for educational purposes as part of a BTech project and is not intended for commercial use.

## Acknowledgements
Special thanks to IIT Roorkee for providing the resources and support necessary to complete this project.
