# SMS Spam Detection

A Flask web application that detects spam SMS messages using TF-IDF and multiple machine-learning classifiers. It includes model evaluation, predictions, MongoDB storage, feedback collection, and database analytics.

## Features

- Spam and ham classification
- TF-IDF text preprocessing
- Grid search model selection
- Accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC metrics
- MongoDB prediction and feedback storage
- Database labels, distributions, trends, and recent activity
- Responsive Flask dashboard

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas and NumPy
- MongoDB and PyMongo
- HTML, CSS, and Jinja

## Project Structure

```text
sms/
├── app.py
├── database_setup.py
├── model_pipeline.py
├── mongodb_service.py
├── spam.csv
├── requirements.txt
├── models/
├── static/
└── templates/
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Shresth71/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start MongoDB

The application expects MongoDB at:

```text
mongodb://localhost:27017
```

MongoDB is optional. The application can still classify messages when MongoDB is unavailable.

## Train the Model

Train and save the model artifacts:

```powershell
python -c "from model_pipeline import train_and_save_model; train_and_save_model()"
```

You can also initialize MongoDB and run demonstration operations:

```powershell
python database_setup.py
```

## Run the Application

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Usage

1. Enter an SMS message in the scanner.
2. Select **Analyze message**.
3. View the predicted label and spam probability.
4. Submit feedback to save the actual label.
5. Use the dashboard to review model metrics and database analytics.

## Model Training

The training pipeline:

- Loads `spam.csv`
- Cleans and normalizes messages
- Removes duplicates and invalid rows
- Converts text to TF-IDF features
- Tests Naive Bayes, Logistic Regression, and Linear SVM candidates
- Selects the best model using cross-validation
- Saves the model and vectorizer in `models/`

## MongoDB Collections

- `messages`: labeled SMS messages
- `predictions`: saved prediction results
- `feedback`: user corrections and feedback

## License

This project is for educational use.
