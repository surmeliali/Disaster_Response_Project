# Disaster Response Pipeline Project


![Screenshot of Web App](screenshots/message0.PNG)


## Table of Contents
1. [Overview](#dependencies)
	* [Dependencies](#dependencies)
	* [Key Files](#key_files)
	* [Executing Program](#execution)
	* [Additional Files](#additional_files)
2. [Installing](#installation)
	* [How to Execute](#execution)
3. [License](#license)
4. [Acknowledgement](#acknowledgement)

## Overview:
The aim is to utilize data engineering and machine learning skills to analyze disaster data from Figure Eight and develop a model for an API that classifies disaster-related messages. The project is structured according to the CRISP-DM (Cross Industry Process for Data Mining) methodology and encompasses three main sections:

1. **Data Processing:** This involves constructing an ETL (Extract, Transform, Load) pipeline to extract data from the provided dataset, clean it, and store it in a SQLite database.
2. **Machine Learning Pipeline:** The data is split into training and testing sets. A machine learning pipeline is then created using NLTK and scikit-learn's Pipeline and GridSearchCV to develop a final model that can classify messages into 36 categories.
3. **Web Development:** A web application is developed to classify messages in real-time.
![Screenshot of Web App](screenshots/overview.PNG)

The machine learning model uses a combination of CountVectorizer and TfidfTransformer to convert text messages into numerical features. It then employs a MultiOutputClassifier for multi-target classification. The model is tuned using GridSearchCV and evaluated for performance, particularly noting that the dataset's imbalance affects accuracy and recall. The final model is saved for deployment in a Dash-based web application, allowing users to input messages and receive real-time classification results, aiding disaster relief efforts by directing messages to the appropriate response agencies.

### Dependencies
* Python 3.x
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="key_files"></a>

### Key Files
- **app/templates/**: HTML template files for the web application.
- **data/process_data.py**: Contains the ETL (Extract, Transform, Load) pipeline for cleaning data, extracting features, and saving the processed data into a SQLite database.
- **models/train_classifier.py**: Implements the machine learning pipeline, which loads the data, trains the model, and saves the trained model as a `.pkl` file for future use.
- **run.py**: Script to launch the Flask web application for classifying disaster-related messages.

<a name="additional_files"></a>

### Additional Files
Within the `data` and `models` directories, you'll find two Jupyter Notebooks that provide a detailed walkthrough of the model's inner workings:

- **ETL Preparation Notebook**: An in-depth guide to the ETL pipeline implementation.
- **ML Pipeline Preparation Notebook**: A comprehensive look at the Machine Learning Pipeline built with NLTK and Scikit-Learn.

The ML Pipeline Preparation Notebook also includes a section dedicated to retraining the model and tuning its parameters using Grid Search.



## Installation:
- Clone the repository:
   `git clone https://github.com/surmeliali/Disaster_Response_Project.git`

<a name="execution"></a>

### How to Execute:
1. Run the following commands in the project's root directory to set up your database and model.

#### ETL pipeline that cleans data and stores in database:
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`


#### ML pipeline that trains classifier and saves:
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your Flask web app.
- `python run.py`


3. Go to http://0.0.0.0:3001/


![Screenshot of Web App](screenshots/results1.PNG)


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Acknowledgement:

I would like to express my gratitude to:

- Udacity for providing the excellent Data Science Nanodegree Program that equipped me with the skills to complete this project.
- Figure Eight for making the valuable dataset available to Udacity for training purposes.