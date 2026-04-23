# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load dataset
df = pd.read_excel('Forecast Data.xlsx')
df.set_index('Week', inplace=True)
coalfcast = df[['Coal_Price']].dropna()  # Drop missing values

# Convert to numpy array
coalfcast_dataset = coalfcast.values

# Apply MinMax Scaling (only on training data)
scaler = MinMaxScaler(feature_range=(0, 1))
training_data_len = int(np.ceil(len(coalfcast_dataset) * 0.90))
train_data = coalfcast_dataset[:training_data_len]  # Only train on 90% of data
scaler.fit(train_data)  # Fit scaler on training data only
coalfcast_scaled_data = scaler.transform(coalfcast_dataset)  # Transform all data

# Define parameters
n_lookback = 52  # 52 weeks (1 year) for training window
n_forecast = 26  # 26 weeks (6 months) forecast

# Prepare training data
x_train, y_train = [], []
for i in range(n_lookback, len(train_data) - n_forecast):
    x_train.append(coalfcast_scaled_data[i - n_lookback:i, 0])  # Input: past 52 weeks
    y_train.append(coalfcast_scaled_data[i:i + n_forecast, 0])  # Output: next 26 weeks

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

# Build LSTM model with Bidirectional LSTM & Dropout
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1))))
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(n_forecast))  # Output layer matches forecast period

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')

# Add callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)

# Train model
model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[early_stopping, reduce_lr], verbose=1)

# Prepare testing data
test_data = coalfcast_scaled_data[training_data_len - n_lookback:, :]
x_test = []
for i in range(n_lookback, len(test_data) - n_forecast):
    x_test.append(test_data[i - n_lookback:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)

# Rescale predictions back to original scale (Fixed Scaling)
predictions_rescaled = scaler.inverse_transform(predictions)

# Prepare validation dataset for visualization
train = coalfcast[:training_data_len]
valid = coalfcast[training_data_len:].copy()

# Adjust prediction alignment
valid['Predictions'] = np.nan
for i in range(len(predictions_rescaled)):
    valid.iloc[i, valid.columns.get_loc('Predictions')] = predictions_rescaled[i, 0]

# Plot actual vs predicted values
plt.figure(figsize=(16, 6))
plt.plot(train['Coal_Price'], label='Train')
plt.plot(valid[['Coal_Price', 'Predictions']], label='Actual vs Predictions')
plt.xlabel('Date')
plt.ylabel('Coal Price (USD/t)')
plt.legend(loc='lower right')
plt.show()

# Forecast 6 months ahead
X_ = coalfcast_scaled_data[-n_lookback:]
X_ = X_.reshape(1, n_lookback, 1)
Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# Prepare forecast DataFrame
df_past = df[['Coal_Price']].reset_index()
df_past['Week'] = pd.to_datetime(df_past['Week'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Coal_Price'].iloc[-1]

df_future = pd.DataFrame(columns=['Week', 'Coal_Price', 'Forecast'])
df_future['Week'] = pd.date_range(start=df_past['Week'].iloc[-1] + pd.Timedelta(days=1),
                                  periods=n_forecast, freq='W-FRI')
df_future['Forecast'] = Y_.flatten()
df_future['Coal_Price'] = np.nan

# Concatenate historical and forecast data
results = pd.concat([df_past, df_future]).set_index('Week')

# Plot forecast
plt.figure(figsize=(16, 6))
plt.plot(results)
plt.title('Coal_Price Price Forecast (12 Months Ahead)')
plt.xlabel('Date')
plt.ylabel('Price (USD/t)')
plt.legend(['Historical', 'Forecast'])
plt.show()

# Display forecast data
df_future

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure valid data for accuracy calculation
valid_actual = valid['Coal_Price'].dropna().values  # Actual prices
valid_predictions = valid['Predictions'].dropna().values  # Model's predicted prices

# Ensure both arrays have the same length
min_len = min(len(valid_actual), len(valid_predictions))
valid_actual = valid_actual[:min_len]
valid_predictions = valid_predictions[:min_len]

# Compute accuracy metrics
mae = mean_absolute_error(valid_actual, valid_predictions)
rmse = np.sqrt(mean_squared_error(valid_actual, valid_predictions))
mape = np.mean(np.abs((valid_actual - valid_predictions) / valid_actual)) * 100

# Print the results
print(f"📊 Model Accuracy Metrics:")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f} USD/t")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f} USD/t")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Ensure error is below 8% threshold
if mape > 8:
    print(f"⚠️ Warning: MAPE is above 8% ({mape:.2f}%) - Consider further tuning.")
else:
    print(f"🎯 Success: MAPE is below 8% ({mape:.2f}%) - Model meets accuracy requirements!")

# Export the result
df_future.to_excel("LSTM result.xlsx")
