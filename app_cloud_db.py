
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, date, time
import requests
import math

from functions_cloud_db import (
    get_city_coordinates,
    get_dataset,
    get_weather,
    get_predictors
)

from graphs import (
    show_hourly_usage,
    show_monthly_trends,
    show_temperature_usage,
    show_city_comparison,
    show_weather_usage
)

# Page configuration
st.set_page_config(
    page_title="CatBoost Prediction",
    page_icon="📊",
    layout="wide"
)

# Load dataset from Railway
df = get_dataset()

# Upload models and categories
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "catboost_model_spline.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "catboost_categories_spline.pkl", "rb") as f:
    categories = pickle.load(f)

# Title
st.title("Welcome to the Bike Sharing Demand Prediction App! 🚲")

tab1, tab2, tab3, tab4 = st.tabs(["Home", 
                                  "Dataset Explorer", 
                                  "Automatic Prediction Tool", 
                                  "Manual Prediction Tool"])

with tab1:
    st.write("""
        ## About the App
        This app predicts hourly bike-sharing demand in Berlin, Dresden, and Cologne based on weather and time.  
        We trained a machine learning model on past bike-sharing data from these cities from years 2022-2023  – on up to 2M trips per city.  
        So now you can estimate the demand for a future date – this way, your company can prepare enough bikes to keep your customers happy!

        ## About the Dataset
        We combined bike-sharing data, weather records, time and calendar data from different sources 
        to analyze and predict hourly bike-sharing demand.

        #### Factors Used for Predictions
        - **Weather Factors:** Overall weather conditions (clear, cloudy, rainy, drizzly, foggy, misty), temperature, humidity, wind speed, amount of rainfall and snowfall in the previous hour.
        - **Time Factors:** Month and hour of the day.
        - **Calendar Factors:** Whether it is working day (vs. a holiday/weekend) and whether it is a school break (Schulferien; vs. regular school time).
        - **Location Factor:** City name (Berlin, Cologne, Dresden).

        #### Target that We Are Predicting
        **Started bike trips:** The number of bike-sharing trips started within each hour. It represents the demand for available ready-to-use bikes that a company needs to prepare.

        #### Data Sources
        - **Bike-Sharing Data:** Historical bike-sharing records obtained
        from the [European Bike Sharing Dataset](https://github.com/tumftm/european-bike-sharing-dataset)¹.
        - **Weather Data:** Weather records collected using the
        [OpenWeather API](https://openweathermap.org/api), including temperature, humidity, wind speed, rainfall,
        and snowfall.
        - **Calendar Data:** Working day information generated using a Python library; school break information taken from the 
        [Schulferien website](https://www.schulferien.org/deutschland/ferien/).
        
        
        ## What Can You Explore?

        #### Data Visualization
        Investigate how weather, time, and calendar factors affect bike-sharing demand in Berlin, Cologne, and Dresden.    

        #### Prediction of Bike-Sharing Demand
        Estimate hourly demand based on weather, time, calendar and location factors outlined above.  
        You have two options here - automatic and manual prediction.

        - **Automatic Prediction**  
        For up to 5 days from now – 
        Enter your desired date and time, and get the predicted demand.    
        The weather forecast will be fetched from [OpenWeather](https://openweathermap.org/api/forecast5?collection=current_forecast)², 
        and all the other factors will be generated automatically.

        - **Manual Prediction**  
        For a later time point or a hypothetical scenario – 
        Enter your desired date and time, along with the weather and calendar factors, and get the predicted demand. 
        Use this option if you want a prediction for a date far in the future for which there is no weather forecast yet, 
        or if you want to see the demand for a hypothetical scenario (a sunny summer weekend or a stormy winter night).

        #### Our GitHub Repository
        To learn more about the data and the machine learning models we used, please visit our [GitHub page](https://github.com/leylabk/Streamlit-app-prediction)!

    """)

    st.caption(
    "¹Dataset source: Waldner et al. (2025), "
    "Data-Driven Insights into (E-)Bike-Sharing, Transportation. "
    "Licensed under CC BY-NC 4.0."
    )
    
    st.caption(
    """²OpenWeather only has 3-hour weather forecasts available to the public for free, while hourly forecasts are behind a paywall. 
    This means that our app has access to only one forecast for a 3-hour window, for example, from 09:00 to 12:00, not one forecast for each hour. 
    This reduces the accuracy of the automatic predictions. 
    If you need an accurate prediction for specific known weather conditions, please use the manual prediction option."""
    )


with tab2:
    st.title("Data Visualization")
    st.markdown(
        "Explore bike-sharing patterns across different "
        "cities, hours, months, and temperature conditions."
    )

    st.divider()

    # First row
    col1, col2 = st.columns(2, gap="large")

    with col1:
        show_hourly_usage(df)

    with col2:
        show_monthly_trends(df)

 
    # Second row
    col3, col4 = st.columns(2, gap="large")

    with col3:
        show_temperature_usage(df)

    with col4:
        show_city_comparison(df)


    # Third row
    col5, col6 = st.columns(2, gap="large")

    show_weather_usage(df)
  
        

with tab3:

    st.header("Automatic Bike Demand Prediction")

    st.caption("""
    - Select a city, date, and time.
    - Weather and calendar information will be retrieved automatically.
    - Weather forecasts are available for the next 5 days. Please select a date within this period.
    """)

    # User selects a city
    city = st.selectbox(
        "Select a city",
        ["Berlin", "Köln", "Dresden"],
        key="automatic_city"
    )

    # User selects a date
    selected_date = st.date_input(
        "Select a date",
        value=date.today(),
        key="automatic_date"
    )

    # User selects a time
    selected_time = st.time_input(
        "Select a time",
        value=time(12, 0),
        key="automatic_time"
    )

    # Combine date and time
    date_time = datetime.combine(
        selected_date,
        selected_time
    )

        # Prediction button
    if st.button(
        "Predict Bike Demand",
        type="primary",
        use_container_width=True,
        key="automatic_predict"
    ):

        # Function 1: Get city coordinates
        coordinates = get_city_coordinates(city)

        if coordinates is None:
            st.error("Coordinates not found for this city.")
            st.stop()

        lat, lon = coordinates

        # Function 2: Get weather forecast
        try:
            selected_forecast = get_weather(
                lat,
                lon,
                date_time,
                city
            )
        except requests.exceptions.RequestException:
            st.error(
                "The weather forecast could not be retrieved. "
                "Please try again later."
            )
            st.stop()

        # Check whether a matching forecast exists
        if selected_forecast.empty:
            st.warning(
                "No weather forecast is available for the "
                "selected date and time. Please choose a "
                "different date or time."
            )
            st.stop()

        # Function 3: Prepare predictors
        X_new = get_predictors(
            selected_forecast,
            date_time
        )

        # Make prediction with the CatBoost model
        prediction = model.predict(X_new)

        # Display prediction
        st.success("Prediction completed!")

        st.metric(
            label="Predicted hourly bike-sharing demand",
            value=f"{math.ceil(max(0, prediction[0])):,} trips"
        )

        # Show model input
        with st.expander("View Input Data"):
            st.dataframe(X_new)



with tab4:
    st.caption(
        "Enter all required input variables manually, "
        "including weather and calendar information, "
        "to predict hourly bike-sharing demand."
    )
# User inputs
    st.subheader("Categorical Features")
    col1, col2 = st.columns(2)

    with col1:
        city_name = st.selectbox(
            "City",
            categories["city_name"]
        )

        weather_group = st.selectbox(
            "Weather Group",
            categories["weather_group"]
        )

        
    with col2:
        working_day = st.selectbox(
                "Working Day",
                categories["working_day"]
            )
        school_break = st.selectbox(
            "School Break",
            categories["school_break"]
        )

        # month = st.selectbox(
        # "Month",
        # categories["month"]
        # )

        # hour = st.selectbox(
        # "Hour",
        # categories["hour"]
        # )

    st.divider()

    st.subheader("Time Features")

    col3, col4 = st.columns(2)

    with col3:
        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1,
            step=1
        )

    with col4:
        hour = st.number_input(
            "Hour",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

    st.divider()

    st.subheader("Weather Measurements")

    col5, col6 = st.columns(2)

    with col5:
        temp = st.number_input("Temperature", value=20.0)
        humidity = st.number_input("Humidity", value=50.0)
        wind_speed = st.number_input("Wind Speed", value=0.0)

    with col6:
        rain_1h = st.number_input("Rain (1h)", value=0.0)
        snow_1h = st.number_input("Snow (1h)", value=0.0)

    st.divider()

    # Prediction
    if st.button("Predict", type="primary", use_container_width=True):

        input_data = pd.DataFrame([{
            "city_name": city_name,
            "weather_group": weather_group,
            "working_day": working_day,
            "school_break": school_break,
            "month": month,
            "hour": hour,
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rain_1h": rain_1h,
            "snow_1h": snow_1h
        }])

        # Make prediction with the CatBoost model
        prediction = model.predict(input_data)

        # Display prediction
        st.success("Prediction completed!")

        st.metric(
            label="Predicted hourly bike-sharing demand",
            value=f"{math.ceil(max(0, prediction[0])):,} trips"
        )

        # Show model input
        with st.expander("View Input Data"):
            st.dataframe(input_data)



