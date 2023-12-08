from flask import Flask, render_template, request
from keras.backend import clear_session
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

clear_session()

app = Flask(__name__)

model = load_model('my_model.h5')
sequence_length = 10
data = pd.read_csv('NSEI_2006.csv', parse_dates=True, index_col='Date')
data.drop(columns=['Open','High','Low','Adj Close','Volume'],inplace=True)
data.rename(columns={'Close':'Nifty'},inplace=True)


@app.route('/',methods=['GET'])
def home():
	return render_template('index.html',pred=' ')

@app.route('/predict',methods=['POST'])
def predict():
	global data
	if request.method == 'POST':
		try:
			 
			future_date_str = request.form['select_date']
			future_date = datetime.strptime(future_date_str, "%Y-%m-%d")

			last_sequence = data['Nifty'].iloc[-sequence_length:].values.reshape(-1, 1)

			scaler = MinMaxScaler(feature_range=(0, 1))
			last_sequence_scaled = scaler.fit_transform(last_sequence)

			last_sequence_scaled = np.reshape(last_sequence_scaled, (1, sequence_length, 1))

			while data.index[-1] < future_date:
				next_prediction_scaled = model.predict(last_sequence_scaled)

				next_prediction = scaler.inverse_transform(next_prediction_scaled)

				last_sequence_scaled = np.roll(last_sequence_scaled, -1)
				last_sequence_scaled[0, -1, 0] = next_prediction_scaled

				data = pd.concat([data,pd.DataFrame({'Nifty': [next_prediction.flatten()[0]]}, index=[data.index[-1] + timedelta(days=1)])],axis=0)

			predicted_value = data.loc[future_date]['Nifty']
			# predicted_value = data.tail()

			pred = f"Stock Prediction value for {future_date_str}: {predicted_value:.2f}"
			return render_template('index.html', pred=pred)

		except Exception as e:
			# Handle any errors that might occur during prediction or date conversion
			error_message = f"Error: {str(e)}"
			return render_template('index.html', error=error_message)

	else:
		return render_template('index.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8080,debug=True)
