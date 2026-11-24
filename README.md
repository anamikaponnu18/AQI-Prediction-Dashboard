# AQI Prediction Dashboard 🌍

## Project Description

The AQI Prediction Dashboard is an interactive, machine learning-based web application developed using Streamlit to predict the next-day Air Quality Index (AQI).

The application uses historical air quality data and pollutant concentrations to estimate future AQI levels using a trained Random Forest Regression model. It provides an interactive interface for monitoring air quality, exploring historical trends, and understanding predicted AQI categories.

## Objectives

- Predict next-day Air Quality Index (AQI) using machine learning.
- Analyze historical air quality patterns and pollutant concentrations.
- Explore the relationship between major air pollutants and AQI.
- Provide an interactive dashboard for visualizing air quality information.
- Display predicted AQI values along with corresponding air quality categories.
- Evaluate and present machine learning model performance.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Development Environment:** Visual Studio Code
- **Deployment:** GitHub and Streamlit Community Cloud

## Machine Learning Model

**Algorithm:** Random Forest Regressor

The model uses 13 input features:

**Pollutant Features**
- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

**Historical AQI Features**
- AQI Lag 1
- AQI Lag 2
- AQI Lag 3
- AQI Lag 7
- 7-Day Average AQI

**Temporal Features**
- Month
- Day of Week

## Project Workflow

1. **Data Collection:** Collect historical air quality data containing pollutant concentrations and AQI measurements.

2. **Data Preprocessing:** Clean the dataset, handle missing values, and prepare the data for further analysis.

3. **Exploratory Data Analysis:** Examine pollutant distributions, AQI variations, and historical air quality patterns.

4. **Feature Engineering:** Generate historical AQI lag features, calculate rolling averages, and extract temporal features.

5. **Model Training:** Train a Random Forest Regressor using the prepared dataset.

6. **Model Evaluation:** Evaluate model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

7. **Model Integration:** Load the trained model and supporting data into the Streamlit application using Joblib.

8. **Dashboard Development:** Create an interactive interface for selecting monitoring stations and viewing pollutant measurements.

9. **AQI Prediction:** Generate next-day AQI predictions using the selected station's historical information and model features.

10. **Visualization:** Display predicted AQI values, air quality categories, historical trends, and model evaluation metrics.

11. **Deployment:** Deploy the Streamlit application using GitHub and Streamlit Community Cloud.

## Key Features

- Monitoring station selection.
- Display of major pollutant concentrations.
- Next-day AQI prediction.
- AQI category classification.
- Historical AQI trend visualization.
- Model performance evaluation metrics.
- Interactive and user-friendly dashboard.

## AQI Categories

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

## Model Evaluation Metrics

The Random Forest model is evaluated using:

- **MAE:** Measures the average absolute difference between actual and predicted AQI values.
- **RMSE:** Measures prediction errors while giving greater weight to larger errors.
- **R² Score:** Indicates how much variation in AQI is explained by the model.

Refer to the model evaluation results for the actual performance values.

## Deployment

The application is designed for deployment using Streamlit Community Cloud, with the project repository hosted on GitHub.

## Disclaimer

This application is developed for academic and educational purposes. The predicted AQI values are estimates and should not be considered substitutes for official air quality monitoring data or public health advisories.

## Acknowledgements

Developed as part of an academic machine learning project focused on air quality prediction and interactive data visualization.
