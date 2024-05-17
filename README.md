Nu Prediction Using ML and DL Models
Overview

This project involves developing an API using Flask to predict Nu (Nusselt number) using Machine Learning (ML) and Deep Learning (DL) models. The following steps will guide you through setting up and running the project locally.
Steps to Run the Project

    Download the Files Locally
        Clone the repository to your local machine using the following command:

        bash

    git clone https://github.com/your-repo/nuprediction.git

Install Required Packages

    Navigate to the project directory:

    bash

cd nuprediction

Install the required packages using pip:

bash

    pip install -r requirements.txt

Run the Application

    Execute the main Python file to start the Flask server:

    bash

    python index.py

Access the API

    Open your browser and navigate to:

    arduino

        http://127.0.0.1:5000/

Project Structure

arduino

nuprediction/
├── models/
│   ├── ml_model.pkl
│   ├── dl_model.h5
├── static/
│   ├── css/
│   ├── js/
├── templates/
│   ├── index.html
├── index.py
├── requirements.txt
├── README.md

Author

    M Hitesh, IIT Roorkee

Notes

    Ensure that Python and pip are installed on your system.
    The requirements.txt file includes all necessary dependencies for the project.
    The ML and DL models are stored in the models directory.
    The main Flask application is in the index.py file.

License

This project is licensed under the MIT License.

Feel free to contact me for any queries or issues regarding this project.
