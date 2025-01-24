# Bankruptcy Prediction Model

This project aims to predict the likelihood of bankruptcy for companies using machine learning techniques. The model is built using Python and various data processing and machine learning libraries.

## Project Structure

- `main.py`: Main script for data processing, model training, and evaluation.
- `requirements.txt`: List of dependencies required for the project.
- `data/`: Directory containing the training and test datasets.
- `outputs/`: Directory where the results and predictions are saved.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ArviHoxha7/Bankruptcy-Prediction-Model.git
   cd Bankruptcy-Prediction-Model
    ```
   
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   
3. Run the main script:
   ```sh
    python main.py
    ```
   
4. Clean generated files:
   ```sh
   rm data/cleaned_training_data.csv outputs/*
   rm -r catboost_info
   ```