# Bike Sharing Demand Forecasting with Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

This project uses a deep learning approach to forecast hourly bike sharing demand. It focuses on building a hybrid **Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)** model to predict the number of `casual` and `registered` bike rentals, leveraging a rich set of temporal and environmental features.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Source](#data-source)
- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Model Variants](#model-variants)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Requirements

This project requires the following Python packages:

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pyyaml>=5.4.0
tqdm>=4.62.0
utilsforecast>=0.0.1
jupyter>=1.0.0
```

**System Requirements:**
- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster training)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dilshan-49/Bike-Sharing-Forecast.git
   cd Bike-Sharing-Forecast
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv bike_forecast_env
   source bike_forecast_env/bin/activate  # On Windows: bike_forecast_env\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib pyyaml tqdm utilsforecast jupyter
   ```

4. **Download the dataset:**
   - Place the `hour.csv` file from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) in the `dataset/` directory

## Data Source

The dataset used for this project is the **Bike Sharing Dataset** from the UCI Machine Learning Repository, based on the Capital Bikeshare system in Washington D.C., USA.

**Dataset Details:**
- **Time Period:** 2011-2012 (2 years of historical data)
- **Records:** 17,379 hourly observations
- **Features:** 16 environmental and temporal features
- **Target Variables:** `casual` and `registered` user counts

**Key Features:**
- **Temporal:** hour, day, month, season, year, weekday, holiday, workingday
- **Weather:** temperature, feeling temperature, humidity, windspeed, weather situation
- **Target:** casual users, registered users, total count

Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

---

## Project Objectives

The primary goals of this project are:
* To preprocess and prepare the Bike Sharing dataset for multivariate time series forecasting.
* To engineer meaningful features that capture seasonal and temporal patterns.
* To design and implement a hybrid CNN-LSTM model, a state-of-the-art architecture for this type of problem.
* To perform a recursive forecast and evaluate the model's performance against actual historical data.

## Project Structure

```
Bike-Sharing-Forecast/
├── README.md                    # Project documentation
├── config.yaml                 # Configuration file for hyperparameters
├── model.py                     # PyTorch model definitions (LSTM, CNN-LSTM)
├── dataset/                     # Data directory
│   ├── hour.csv                # Original UCI dataset
│   ├── train.csv               # Training set (processed)
│   ├── val.csv                 # Validation set (processed)
│   ├── test.csv                # Test set (processed)
│   └── Readme.txt              # Dataset documentation
├── models/                      # Saved model checkpoints
│   ├── cnn_lstm_*_best.pth     # CNN-LSTM model variants
│   └── lstm_*_best.pth         # LSTM model variants
├── Images/                      # Result visualizations
│   ├── recursive forecasting.png
│   ├── full forecast.png
│   └── model performance plots
├── Notebooks/
│   ├── process_data.ipynb      # Data preprocessing and feature engineering
│   ├── model_training.ipynb    # Model training pipeline
│   ├── model_testing.ipynb     # Model evaluation and testing
│   └── recursive_forecast.ipynb # Multi-step recursive forecasting
└── 7 param CNN-LSTM.png        # Model architecture diagram
```

### Core Components

* **`process_data.ipynb`**: Comprehensive data preprocessing pipeline that handles outlier detection, missing values, feature engineering (temporal features with sine/cosine transformations), and chronological data splitting.

* **`model.py`**: Contains PyTorch model definitions including:
  - `LSTMForecaster`: Pure LSTM architecture for baseline comparison
  - `CNN_LSTMForecaster`: Hybrid CNN-LSTM architecture for capturing both local patterns and long-term dependencies

* **`model_training.ipynb`**: Training pipeline with configurable hyperparameters, L1Loss optimization, early stopping, and model checkpointing based on validation performance.

* **`model_testing.ipynb`**: Comprehensive model evaluation including performance metrics, visualization of predictions vs actual values, and comparison between model variants.

* **`recursive_forecast.ipynb`**: Demonstrates the model's primary application using recursive, walk-forward forecasting strategy for multi-step predictions.

* **`config.yaml`**: Centralized configuration management for hyperparameters, feature selection, and model architecture parameters.

## Configuration

The project uses a YAML configuration file (`config.yaml`) for easy parameter management:

```yaml
dataset:
  lookback_window: 168     # 7 days * 24 hours
  forecast_horizon: 24     # 24 hours ahead
  shift: 1                 # Step size for sliding window
  batch_size: 32           # Training batch size

data:
  feature_columns:         # Selected features for training
    - workingday
    - day_sin             # Sine transformation of hour
    - day_cos             # Cosine transformation of hour  
    - week_sin            # Sine transformation of weekday
    - week_cos            # Cosine transformation of weekday
    - casual_s            # Scaled casual user count
    - registered_s        # Scaled registered user count
  target_columns:
    - casual_s
    - registered_s

model:
  num_features: 7          # Number of input features
  num_targets: 2           # Number of target variables
```

**Key Configuration Parameters:**
- **lookback_window**: Historical time steps used for prediction (168 = 1 week)
- **forecast_horizon**: Number of future time steps to predict (24 = 1 day)
- **feature_columns**: Specific features used for training (can be modified for experiments)
- **Temporal Features**: Sine/cosine transformations capture cyclical patterns without introducing artificial ordering

## Methodology

### 1. Data Preprocessing & Feature Engineering

The `process_data.ipynb` notebook prepares the raw dataset for the deep learning model.

* **Handling Categorical Features:** Features like `season`, `workingday`, and `weathersit` are converted using **one-hot encoding** to avoid introducing an artificial ordinal relationship.
* **Scaling:** All numerical features and the targets are scaled using `MinMaxScaler` to a consistent range of `[0, 1]`, which is essential for deep learning models.
* **Temporal Features:** To capture daily and weekly seasonality, new features were engineered using sine and cosine transformations of the hour and weekday. For example:  
    * $Daily_{sin} = sin(2 \pi \times \frac{hour}{24})$
    * $Weekly_{cos} = cos(2 \pi \times \frac{day\_of\_week}{7})$

### 2. Model Architecture

The core of this project is a hybrid **CNN-LSTM** architecture designed to capture both short-term and long-term dependencies in the data.

* **Input Sequence:** The model takes a historical sequence with a **168-hour (7-day)** lookback window. Crucially, this input sequence includes the past values of the `casual` and `registered` counts, making the model **autoregressive**.
* **CNN Layer:** A 1D Convolutional layer acts as a feature extractor, identifying local, recurring patterns within the historical sequence (e.g., daily rush hour spikes).
* **LSTM Layer:** The output from the CNN is fed into an LSTM layer. The LSTM's recurrent nature allows it to remember information over long sequences, which is crucial for capturing yearly and weekly trends.
* **Output Layer:** A final linear layer maps the LSTM's learned representation to the desired **24-hour** forecast block in a single forward pass.

## Model Variants

The project includes multiple model architectures and configurations:

### Architecture Types
1. **LSTM-only (`LSTMForecaster`)**: Pure LSTM architecture for baseline comparison
2. **CNN-LSTM (`CNN_LSTMForecaster`)**: Hybrid architecture combining CNN feature extraction with LSTM temporal modeling

### Feature Configurations
The models are trained with different feature sets (indicated by the number in model filenames):
- **4 features**: Basic temporal features only
- **7 features**: Enhanced feature set (current configuration)
- **9 features**: Extended feature set including weather variables

### Model Selection Guidelines
- **LSTM**: Better performance with shorter lookback windows and smaller forecast horizons
- **CNN-LSTM**: Superior performance with longer lookback windows (168h) and larger forecast horizons (24h)
- **Feature Selection**: 7-feature configuration provides optimal balance between complexity and performance

## Results

### Model Performance
The hybrid CNN-LSTM model demonstrates superior performance compared to the LSTM-only baseline:

![Model Architecture](7%20param%20CNN-LSTM.png)

**Key Performance Insights:**
- CNN-LSTM outperforms LSTM for long-term forecasting (24-hour horizon)
- L1Loss (Mean Absolute Error) provides robustness against outliers
- Recursive forecasting maintains accuracy over extended prediction periods

### Visualizations Available
The project includes several visualization outputs in the `Images/` directory:
- **Recursive Forecasting**: Shows multi-step prediction capability
- **Full Forecast**: Complete test set predictions vs actual values
- **Model Comparisons**: Performance across different architectures and feature sets

![Recursive Forecasting Results](Images/recursive%20forecasting.png)

### Performance Metrics
- Training uses L1Loss (Mean Absolute Error) for robust optimization
- Validation-based early stopping prevents overfitting
- Walk-forward validation ensures temporal consistency

## How to Run the Project

### Quick Start
1. **Setup Environment:**
   ```bash
   git clone https://github.com/dilshan-49/Bike-Sharing-Forecast.git
   cd Bike-Sharing-Forecast
   pip install torch pandas numpy scikit-learn matplotlib pyyaml tqdm utilsforecast jupyter
   ```

2. **Prepare Dataset:**
   - Download `hour.csv` from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
   - Place it in the `dataset/` directory

3. **Run the Complete Pipeline:**
   Execute the Jupyter notebooks in the following order:
   
   **Step 1: Data Preprocessing**
   ```bash
   jupyter notebook process_data.ipynb
   ```
   - Loads raw dataset and handles missing values
   - Engineers temporal features (sine/cosine transformations)
   - Splits data chronologically into train/validation/test sets
   - Applies MinMax scaling

   **Step 2: Model Training**
   ```bash
   jupyter notebook model_training.ipynb
   ```
   - Trains CNN-LSTM model with configurable hyperparameters
   - Uses L1Loss and early stopping based on validation performance
   - Saves best model checkpoint

   **Step 3: Model Evaluation**
   ```bash
   jupyter notebook model_testing.ipynb
   ```
   - Loads trained model and evaluates on test set
   - Generates performance metrics and visualizations
   - Compares different model variants

   **Step 4: Recursive Forecasting**
   ```bash
   jupyter notebook recursive_forecast.ipynb
   ```
   - Demonstrates multi-step forecasting capability
   - Uses walk-forward validation approach
   - Generates forecast visualizations

### Alternative: Command Line Execution
For automated execution without Jupyter interface:
```bash
jupyter nbconvert --execute --to notebook process_data.ipynb
jupyter nbconvert --execute --to notebook model_training.ipynb
jupyter nbconvert --execute --to notebook model_testing.ipynb
jupyter nbconvert --execute --to notebook recursive_forecast.ipynb
```

## Advanced Usage

### Custom Configuration
Modify `config.yaml` to experiment with different settings:

```yaml
# Experiment with different lookback windows
dataset:
  lookback_window: 336  # 2 weeks instead of 1 week
  forecast_horizon: 48  # 2 days instead of 1 day

# Try different feature combinations
data:
  feature_columns:
    - workingday
    - temp              # Add temperature
    - hum               # Add humidity
    - day_sin
    - day_cos
    - week_sin
    - week_cos
    - casual_s
    - registered_s
```

### Model Architecture Customization
Modify model parameters in the training notebook:

```python
# CNN-LSTM with custom parameters
model = CNN_LSTMForecaster(
    num_features=config['model']['num_features'],
    hidden_size=128,        # Increase hidden size
    num_layers=3,           # Add more LSTM layers
    output_size=forecast_horizon * num_targets,
    cnn_filters=128,        # More CNN filters
    kernel_size=5           # Different kernel size
)
```

### Recursive Forecasting Strategy
The recursive approach enables multi-step forecasting:

1. **Initial Prediction**: Use historical data to predict next 24 hours
2. **Update Input**: Replace oldest 24 hours with predictions
3. **Next Prediction**: Generate forecast for subsequent 24 hours
4. **Repeat**: Continue for entire test period

This approach tests the model's ability to maintain accuracy when using its own predictions as input.

### Performance Monitoring
Track training progress with custom metrics:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate additional metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
```

## Contributing

We welcome contributions to improve the project! Here's how you can help:

### Types of Contributions
- **Bug Fixes**: Report and fix issues
- **Feature Enhancements**: Add new model architectures or evaluation metrics
- **Documentation**: Improve documentation and examples
- **Performance Optimization**: Speed up training or inference

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with clear description

### Contribution Guidelines
- Follow existing code style and structure
- Add tests for new functionality
- Update documentation for any changes
- Ensure notebooks run without errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Dataset License
The bike sharing dataset usage must cite the original publication:

> Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

### Acknowledgments
- UCI Machine Learning Repository for providing the dataset
- Capital Bikeshare (Washington D.C.) for the original data collection
- PyTorch team for the deep learning framework