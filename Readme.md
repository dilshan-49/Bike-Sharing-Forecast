# Bike Sharing Demand Forecasting with Deep Learning

This project uses a deep learning approach to forecast hourly bike sharing demand. It focuses on building a hybrid **Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)** model to predict the number of `casual` and `registered` bike rentals, leveraging a rich set of temporal and environmental features.

## Data Source

The dataset used for this project is the Bike Sharing Dataset, which is publicly available from the UCI Machine Learning Repository.

Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

---

## Project Objectives

The primary goals of this project are:
* To preprocess and prepare the Bike Sharing dataset for multivariate time series forecasting.
* To engineer meaningful features that capture seasonal and temporal patterns.
* To design and implement a hybrid CNN-LSTM model, a state-of-the-art architecture for this type of problem.
* To perform a recursive forecast and evaluate the model's performance against actual historical data.

## Project Structure

The project is organized into a series of Jupyter notebooks and a core Python script.

* `process_data.ipynb`: This notebook handles all data preprocessing. It loads the raw data, handles outliers and missing values, creates time-based features, and splits the data chronologically into training, validation, and testing sets.
* `model.py`: This script contains the PyTorch model definitions, including the `LSTMForecaster` and `CNN_LSTMForecaster` classes. It serves as the core of the model's architecture.
* `model_training.ipynb`: This notebook uses the preprocessed data to train the CNN-LSTM model. It defines the training loop, uses `L1Loss` (Mean Absolute Error), and saves the best-performing model based on validation loss.
* `model_testing.ipynb`: This notebook loads a trained model and evaluates its performance on the test set. It includes a custom function to filter and plot non-overlapping predictions to provide an intuitive visualization of the model's performance.
* `recursive_forecast.ipynb`: This notebook demonstrates the model's primary application. It uses a recursive, walk-forward strategy to generate multi-step forecasts for the entire test set.
* `config.yaml`: A configuration file for managing hyperparameters such as lookback window size and feature columns, ensuring a clean separation of code and configuration.

## Methodology

### 1. Data Preprocessing & Feature Engineering

The `process_data.ipynb` notebook prepares the raw dataset for the deep learning model.

* **Handling Categorical Features:** Features like `season`, `workingday`, and `weathersit` are converted using **one-hot encoding** to avoid introducing an artificial ordinal relationship.
* **Handling Skewness:** Highly skewed columns, including the `casual` and `registered` user counts, are **log-transformed** to stabilize variance and improve model convergence.
* **Scaling:** All numerical features and the log-transformed targets are scaled using `MinMaxScaler` to a consistent range of `[0, 1]`, which is essential for deep learning models.
* **Temporal Features:** To capture daily and weekly seasonality, new features were engineered using sine and cosine transformations of the hour and weekday. For example:  
    * $Daily_{sin} = sin(2 \pi \times \frac{hour}{24})$
    * $Weekly_{cos} = cos(2 \pi \times \frac{day\_of\_week}{7})$

### 2. Model Architecture

The core of this project is a hybrid **CNN-LSTM** architecture designed to capture both short-term and long-term dependencies in the data.

* **Input Sequence:** The model takes a historical sequence with a **168-hour (7-day)** lookback window. Crucially, this input sequence includes the past values of the `casual` and `registered` counts, making the model **autoregressive**.
* **CNN Layer:** A 1D Convolutional layer acts as a feature extractor, identifying local, recurring patterns within the historical sequence (e.g., daily rush hour spikes).
* **LSTM Layer:** The output from the CNN is fed into an LSTM layer. The LSTM's recurrent nature allows it to remember information over long sequences, which is crucial for capturing yearly and weekly trends.
* **Output Layer:** A final linear layer maps the LSTM's learned representation to the desired **24-hour** forecast block in a single forward pass.

## Results & Conclusion

The model was trained using `L1Loss` (Mean Absolute Error) to make it more robust to outliers. The performance was evaluated using a recursive forecasting strategy. This approach demonstrated the model's ability to not only predict a single future point but to continuously forecast for a full day by feeding its own predictions back into the input. The model's strong performance on unseen data highlights the power of deep learning for complex, multi-variate time series forecasting problems.

## How to Run the Project

1. Place your dataset file (`hour.csv`) in a directory named `dataset`.
2. Ensure you have all necessary Python libraries installed.
3. Run the Jupyter notebooks in the following order to execute the full pipeline:
    1. `process_data.ipynb`
    2. `model_training.ipynb`
    3. `model_testing.ipynb`
    4. `recursive_forecast.ipynb`