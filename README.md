# Stock Prediction with LSTM and Flask

## Overview
This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network trained on historical data from the Nifty dataset. The model is deployed using Flask, allowing users to interact with the prediction system through a web interface.

## Dataset
- **Source:** Obtained from the Nifty website
- **Records:** 4677
- **Features:**
  - Date
  - Nifty (stock prices)

## Project Structure
- **Modeling:** LSTM is used to build a predictive model on historical stock data.
- **Deployment:** The Flask web application allows users to input a date and receive a predicted Nifty value.
- **Web Interface:** The web interface (`templates/index.html`) provides a user-friendly way to interact with the prediction model.

## Files and Directories
- **app.py:** Flask application code for handling user requests and rendering the web interface.
- **templates/index.html:** HTML template for the web interface.
- **stockpredictionLSTM.ipynb:** Jupyter Notebook containing the LSTM model development and training.
- **requirements.txt:** Dependencies required for running the Flask application.

## Dependencies
- Python 3.x
- Flask
- TensorFlow
- Pandas
- Numpy
- Matplotlib

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the web interface at `http://localhost:5000` in your web browser.
5. Enter a date to get the predicted Nifty value.

## Model Training
- The LSTM model is developed and trained in the `stockpredictionLSTM.ipynb` notebook.
- Modify hyperparameters or architecture as needed for experimentation.

## Acknowledgments
- Nifty dataset: [Nifty Website](https://example.com/nifty-dataset)

## License
This project is licensed under the [MIT License](LICENSE).

## Author
Aswathi
