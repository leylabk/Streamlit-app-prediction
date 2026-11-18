
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, date, time, timedelta
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

from functions_cloud_db import (
    parse_user_date,
    parse_user_time
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

import base64


with open("bike.png", "rb") as image_file:
    bike_image = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
<div style="
    background-color: #EFF7EC;
    background-image: url('data:image/png;base64,{bike_image}');
    background-repeat: no-repeat;
    background-position: right 25px center;
    background-size: 160px auto;
    background-blend-mode: multiply;
    padding: 35px 240px 35px 30px;
    border-radius: 15px;
    min-height: 180px;
">
    <h1 style="color: #285B38;">
        Welcome to BikeCast!
    </h1>
    <p style="color: #45634C; font-size: 17px;">
        Predict hourly bike-sharing demand in
        Berlin, Dresden and Cologne.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Homepage",
    "Data Visualization",
    "Automatic Prediction",
    "Manual Prediction",
    "Explore BikeCast"
])

with tab1:
    st.write("""
        ## About the App
        This app predicts hourly bike-sharing demand in Berlin, Dresden, and Cologne based on weather and time.  
        We trained a machine learning model on past bike-sharing data from these cities from years 2022-2023  – on up to 2M trips per city.  
        So now you can estimate the demand for a future date – this way, your company can prepare enough bikes to keep your customers happy!

        #### Model Performance
        Our machine learning model (**CatBoost**, a gradient boosting algorithm) was evaluated on unseen historical data
        to assess how accurately it predicts hourly bike-sharing demand.
        
        - **R²:** Our model explains 93% of the variation in hourly bike rentals. 
        This means it captures most of the patterns behind why the number of bike rentals changes from one hour to another.
        The remaining 7% may be influenced by other factors that are not included in our model.
       
        ## About the Dataset
        We combined bike-sharing data, weather records, time and calendar data from different sources 
        to analyze and predict hourly bike-sharing demand.

        #### Target that We Are Predicting
        **Started bike trips:** The number of bike-sharing trips started within each hour. It represents the demand for available ready-to-use bikes that a company needs to prepare.

        #### Factors Used for Predictions
        - **Location and Calendar Factors:** City (Berlin, Cologne, Dresden), month, hour of the day, 
        working day (vs. holiday/weekend), 
        and school holidays (Schulferien vs. regular school periods).
        - **Weather Factors:** Weather conditions (clouds, clear, rain, snow, mist, drizzle or fog), 
        temperature, humidity, wind speed, 
        rainfall and snowfall during the previous hour.

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
        
        #### Meet the Team
        Interested in learning more about the people behind this project?
        Connect with us on LinkedIn!

        - Tatiana Pashkova <a href="https://www.linkedin.com/in/tatiana-pashkova/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="18" style="filter: invert(27%) sepia(91%) saturate(1600%) hue-rotate(184deg) brightness(95%) contrast(95%); vertical-align: middle;"></a>
        - Leyla Bakacak Karabenli <a href="https://www.linkedin.com/in/leyla-bakacak-karabenli/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="18" style="filter: invert(27%) sepia(91%) saturate(1600%) hue-rotate(184deg) brightness(95%) contrast(95%); vertical-align: middle;"></a>
        """, unsafe_allow_html=True)

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
    st.markdown("""
        Explore bike-sharing demand across different cities, hours, months, and weather conditions.        
        Bike-sharing demand means the number of started bike-sharing trips within each hour.
        """
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

    st.header("Bike Demand Prediction")

    st.caption(
        "Predict hourly bike-sharing demand using "
        "automatic weather forecasts and calendar information."
    )


    # STEP 1: INDIVIDUAL PREDICTION
    with st.container(border=True):

        st.subheader("1. Individual Bike Demand Prediction")

        st.caption(
            "Select a city, date and time to predict "
            "bike-sharing demand for a single time point."
        )
        st.info(
            "Weather forecast data is available for the next 5 days (120 hours)."
        )

        city = st.selectbox(
            "Select a city",
            ["Berlin", "Köln", "Dresden"],
            key="automatic_city"
        )

        selected_date = st.date_input(
            "Select a date",
            value=date.today(),
            key="automatic_date"
        )

        selected_time = st.time_input(
            "Select a time",
            value=time(12, 0),
            key="automatic_time"
        )

        date_time = datetime.combine(
            selected_date,
            selected_time
        )

        if st.button(
            "Generate Predictions",
            type="primary",
            use_container_width=True,
            key="automatic_predict"
        ):

            try:

                coordinates = get_city_coordinates(city)

                if coordinates is None:
                    st.error("Coordinates not found.")
                    st.stop()

                lat, lon = coordinates

                selected_forecast = get_weather(
                    lat,
                    lon,
                    date_time,
                    city
                )

                if selected_forecast.empty:
                    st.warning(
                        "No weather forecast available "
                        "for the selected date and time."
                    )
                    st.stop()

                X_new = get_predictors(
                    selected_forecast,
                    date_time
                )

                prediction = model.predict(X_new)

                predicted_demand = math.ceil(
                    max(0, prediction[0])
                )

                st.success("Prediction completed!")

                st.metric(
                    label="Predicted hourly bike-sharing demand",
                    value=f"{predicted_demand:,} trips"
                )

                # Prepare results including weather data

                weather_row = selected_forecast.iloc[0]

                prediction_results = pd.DataFrame([{
                    "city": city,
                    "date": selected_date,
                    "time": selected_time,
                    "temperature_C": weather_row["temp"],
                    "humidity_%": weather_row["humidity"],
                    "wind_speed_m_s": weather_row["wind_speed"],
                    "weather": weather_row["weather_group"],
                    "rain_1h_mm": weather_row["rain_1h"],
                    "snow_1h_mm": weather_row["snow_1h"],
                    "working_day": X_new.iloc[0]["working_day"],
                    "school_break": X_new.iloc[0]["school_break"],
                    "predicted_bike_demand": predicted_demand
                }])

                csv = prediction_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction (CSV)",
                    data=csv,
                    file_name=(
                        f"bike_demand_prediction_{city}_"
                        f"{selected_date}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

                with st.expander("View Input and Prediction Data"):
                    st.dataframe(
                        prediction_results,
                        use_container_width=True
                    )

            except requests.exceptions.RequestException:
                st.error(
                    "The weather forecast could not be retrieved. "
                    "Please try again later."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # STEP 2: BATCH PREDICTION

    with st.container(border=True):

        st.subheader("2. Batch Bike Demand Prediction")

        st.caption(
            "Upload a CSV file containing multiple cities, "
            "dates and times to automatically predict "
            "bike-sharing demand for all entries."
        )

        # CSV TEMPLATE

        template_df = pd.DataFrame({
            "city": ["Berlin", "Köln", "Dresden"],
            "date": [
                "2026-11-17",
                "2026-11-17",
                "2026-11-18"
            ],
            "time": ["08:00", "12:00", "17:00"]
        })

        template_csv = template_df.to_csv(
            index=False
        ).encode("utf-8")

        # DOWNLOAD AND UPLOAD SIDE BY SIDE

        st.markdown("""
        <style>

        /* Download button */
        div[data-testid="stDownloadButton"] button {
            width: 100%;
            height: 48px;
            border: 1px solid #808080;
            border-radius: 8px;
            background-color: transparent;
            font-weight: 600;
        }

        /* File uploader outer border */
        div[data-testid="stFileUploader"] section {
            width: 100% !important;
            min-height: 48px !important;
            height: 48px !important;
            box-sizing: border-box !important;
            padding: 0 12px !important;
            border: 1px solid #808080 !important;
            border-radius: 8px !important;
            background-color: transparent !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* Hide drag and drop instructions */
        div[data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }

        /* Upload button */
        div[data-testid="stFileUploader"] section button {
            height: 36px !important;
            border-radius: 6px !important;
        }

        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Download CSV Template**")
            st.caption("Download the example entries.")

            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="bike_demand_batch_template.csv",
                mime="text/csv",
                use_container_width=True,
                key="manual_batch_template_download"
            )

        with col2:
            st.markdown("**Upload Input Data**")
            st.caption("Upload your completed CSV file.")

            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                key="manual_batch_csv_upload",
                label_visibility="collapsed"
            )

        st.info(
            "Replace the example rows with your own entries."
        )

        st.caption(
            """
            **Required columns:**
            - city: Berlin, Köln or Dresden
            - date: YYYY-MM-DD, DD.MM.YYYY or DD/MM/YYYY
            - time: HH:MM (24-hour format)
            """
        )

        # READ CSV

        if uploaded_file is not None:

            try:

                batch_data = pd.read_csv(
                    uploaded_file,
                    encoding="utf-8-sig",
                    dtype=str,
                    keep_default_na=False
                )

            except UnicodeDecodeError:

                uploaded_file.seek(0)

                try:
                    batch_data = pd.read_csv(
                        uploaded_file,
                        encoding="cp1252",
                        dtype=str,
                        keep_default_na=False
                    )

                except Exception:
                    st.error("The CSV file could not be read.")
                    st.stop()

            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
                st.stop()

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.stop()

            # VALIDATION

            if batch_data.empty:
                st.warning("The uploaded CSV contains no entries.")
                st.stop()

            required_columns = ["city", "date", "time"]

            missing_columns = [
                col for col in required_columns
                if col not in batch_data.columns
            ]

            if missing_columns:
                st.error(
                    f"Missing columns: {missing_columns}"
                )
                st.stop()

            for col in required_columns:
                batch_data[col] = batch_data[col].str.strip()

            if (batch_data[required_columns] == "").any().any():
                st.error(
                    "Some rows contain missing values."
                )
                st.stop()

            valid_cities = ["Berlin", "Köln", "Dresden"]

            invalid_cities = batch_data[
                ~batch_data["city"].isin(valid_cities)
            ]

            if not invalid_cities.empty:
                st.error("Invalid city names found.")
                st.dataframe(invalid_cities[["city"]])
                st.stop()

            # DATE AND TIME VALIDATION

            parsed_dates = []
            parsed_times = []
            validation_errors = []

            for index, row in batch_data.iterrows():

                try:

                    parsed_date = parse_user_date(row["date"])
                    parsed_time = parse_user_time(row["time"])

                    parsed_dates.append(parsed_date)
                    parsed_times.append(parsed_time)

                except ValueError as e:

                    validation_errors.append(
                        f"Row {index + 2}: {e}"
                    )

                    parsed_dates.append(None)
                    parsed_times.append(None)

            if validation_errors:

                st.error("Invalid dates or times:")

                for error in validation_errors:
                    st.write(f"- {error}")

                st.stop()

            batch_data["date"] = parsed_dates
            batch_data["time"] = parsed_times

            batch_data["date_time"] = batch_data.apply(
                lambda row: datetime.combine(
                    row["date"],
                    row["time"]
                ),
                axis=1
            )

            # FORECAST DATE VALIDATION

            today = date.today()
            max_forecast_date = today + timedelta(days=5)

            invalid_dates = batch_data[
                (batch_data["date"] < today) |
                (batch_data["date"] > max_forecast_date)
            ]

            if not invalid_dates.empty:

                st.error(
                    "Some dates are outside the supported "
                    "forecast period."
                )

                st.dataframe(
                    invalid_dates[required_columns],
                    use_container_width=True
                )

                st.stop()

            st.success(
                "CSV file uploaded and validated successfully!"
            )

            # BATCH PREDICTION

            if st.button(
                "Generate Predictions",
                type="primary",
                use_container_width=True,
                key="manual_automatic_batch_predict"
            ):

                results = []
                errors = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                total_rows = len(batch_data)

                for index, row in batch_data.iterrows():

                    city = row["city"]
                    date_time = row["date_time"]

                    status_text.write(
                        f"Processing {index + 1} of {total_rows}: "
                        f"{city} - {date_time}"
                    )

                    try:

                        coordinates = get_city_coordinates(city)

                        if coordinates is None:
                            raise ValueError(
                                "Coordinates not found."
                            )

                        lat, lon = coordinates

                        selected_forecast = get_weather(
                            lat,
                            lon,
                            date_time,
                            city
                        )

                        if selected_forecast.empty:
                            raise ValueError(
                                "No weather forecast available."
                            )

                        X_new = get_predictors(
                            selected_forecast,
                            date_time
                        )

                        prediction = model.predict(X_new)

                        predicted_demand = math.ceil(
                            max(0, prediction[0])
                        )

                        weather_row = selected_forecast.iloc[0]

                        result = {
                            "city": city,
                            "date": row["date"],
                            "time": row["time"],
                            "temperature_C": weather_row["temp"],
                            "humidity_%": weather_row["humidity"],
                            "wind_speed_m_s": weather_row["wind_speed"],
                            "weather": weather_row["weather_group"],
                            "rain_1h_mm": weather_row["rain_1h"],
                            "snow_1h_mm": weather_row["snow_1h"],
                            "predicted_bike_demand": predicted_demand
                        }

                        results.append(result)

                    except requests.exceptions.RequestException:

                        errors.append({
                            "city": city,
                            "date": row["date"],
                            "time": row["time"],
                            "error": "Weather API request failed."
                        })

                    except Exception as e:

                        errors.append({
                            "city": city,
                            "date": row["date"],
                            "time": row["time"],
                            "error": str(e)
                        })

                    progress_bar.progress(
                        (index + 1) / total_rows
                    )

                status_text.empty()
                progress_bar.empty()

                # RESULTS

                if results:

                    results_df = pd.DataFrame(results)

                    st.success(
                        f"{len(results)} predictions completed successfully!"
                    )

                    st.markdown("**Prediction Results**")

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        height=200
                    )

                    results_csv = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download All Predictions (CSV)",
                        data=results_csv,
                        file_name="bike_demand_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="manual_batch_results_download"
                    )

                if errors:

                    errors_df = pd.DataFrame(errors)

                    st.warning(
                        f"{len(errors)} entries could not be predicted."
                    )

                    with st.expander("View Failed Entries"):

                        st.dataframe(
                            errors_df,
                            use_container_width=True,
                            height=200
                        )

                        errors_csv = errors_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Failed Entries (CSV)",
                            data=errors_csv,
                            file_name="bike_demand_failed_entries.csv",
                            mime="text/csv",
                            key="manual_batch_errors_download"
                        )



with tab4:
    st.header("Manual Bike Demand Prediction")

    st.caption(
        "Enter prediction variables manually or upload a CSV file "
        "to generate multiple hourly bike-sharing demand predictions."
    )

    with st.expander("📘 Input Guidelines – Data Description and Accepted Values"):
    
        st.markdown("""
        <div style="font-size: 14px;">
    
        Please follow these guidelines when entering prediction data or preparing
        your CSV file for Batch Prediction.
    
        ### 1. Location and Calendar Information
    
        **City (`city`)**
        - Berlin, Dresden or Cologne
        
        **Month (`month`)**
        - Whole numbers from 1 to 12 (January = 1, December = 12)
    
        **Hour (`hour`)**
        - Whole numbers from 0 to 23 (Midnight = 0, noon = 12, 5 PM = 17)   
        
        **Working Day (`workingday`)**
        - `0` – Holiday or weekend
        - `1` – Working day
    
        **School Break (`school_break`)**
        - `0` – Regular school days
        - `1` – School holidays (Schulferien)
    
    
        ### 2. Weather Information
        **Weather Conditions (`weather_group`)**
        - Clouds, Clear, Rain, Snow, Mist, Drizzle or Fog
        
        **Temperature (`temperature`, °C)**
        - Numbers with up to two decimal places
        - Typical July noon average: 23–26 °C
        - Typical January noon average: 5–6 °C
    
        **Humidity (`humidity`, %)**
        - Whole numbers from 0 to 100
        - Typical average humidity: 72–77%
    
        **Wind Speed (`windspeed`, m/s)**
        - Numbers with up to two decimal places
        - Calm: 0.0–0.2 m/s
        - Moderate breeze: 5.5–7.9 m/s
        - Strong breeze: 10.8–13.8 m/s
    
        **Rain (`rain`, mm)**
        - Rainfall during the previous hour
        - Very light: 0.25–1.0 mm
        - Moderate: 2.5–10 mm
        - Heavy: above 10 mm
    
        **Snow (`snow`, mm)**
        - Snowfall during the previous hour
        - Light: 0.25–1.0 mm
        - Moderate: 1.0–2.5 mm
        - Heavy: above 2.5 mm
    
        ---
    
        **Important**
        - Use the exact column names and formats in the downloadable CSV template.
        - Weather measurements should represent realistic conditions for the selected city, date and hour.
        - The values above are examples, not strict input limits.
    
        </div>
        """, unsafe_allow_html=True)
    
        st.caption(
            "Sources: "
            "[Beaufort Wind Scale](https://www.meteoswiss.admin.ch/dam/jcr:c1064242-4a6e-4df2-8d21-96e68d1def7a/Beaufort_Tabelle_en.pdf) · "
            "[Rainfall Guide](https://rainsimulator.com/guides/intensity-categories) · "
            "[Snowfall Guide](https://www.foxweather.com/learn/snow-101-the-basics-of-snow)"
        )

    # STEP 1: INDIVIDUAL PREDICTION
    with st.container(border=True):
        st.subheader("1. Individual Bike Demand Prediction")

        st.caption(
            "Enter all required input variables manually, "
            "including weather and calendar information."
        )

        st.markdown("**Location and Calendar Factors**")

        col1, col2 = st.columns(2)

        with col1:
            city_name = st.selectbox(
                "City",
                categories["city_name"],
                key="manual_city"
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=1,
                step=1,
                key="manual_month"
            )

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1,
                key="manual_hour"
            )

        with col2:
            working_day = st.selectbox(
                "Working Day",
                categories["working_day"],
                key="manual_working_day"
            )

            school_break = st.selectbox(
                "School Break",
                categories["school_break"],
                key="manual_school_break"
            )

        st.divider()

        st.markdown("**Weather Factors**")

        col3, col4 = st.columns(2)

        with col3:
            weather_group = st.selectbox(
                "Weather Conditions",
                categories["weather_group"],
                key="manual_weather"
            )

            temp = st.number_input(
                "Temperature",
                value=20.0,
                key="manual_temp"
            )

            humidity = st.number_input(
                "Humidity",
                value=50.0,
                key="manual_humidity"
            )

        with col4:
            wind_speed = st.number_input(
                "Wind Speed",
                value=0.0,
                key="manual_wind"
            )
            rain_1h = st.number_input(
                "Rain (1h)",
                value=0.0,
                key="manual_rain"
            )

            snow_1h = st.number_input(
                "Snow (1h)",
                value=0.0,
                key="manual_snow"
            )

        if st.button(
            "Generate Predictions",
            type="primary",
            use_container_width=True,
            key="individual_manual_predict"
        ):

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

            prediction = model.predict(input_data)

            predicted_demand = math.ceil(
                max(0, prediction[0])
            )

            st.success("Prediction completed!")

            st.metric(
                label="Predicted Hourly Bike-Sharing Demand",
                value=f"{predicted_demand:,} trips"
            )

            prediction_results = input_data.copy()

            prediction_results["predicted_bike_demand"] = (
                predicted_demand
            )

            st.markdown("**Prediction Results**")

            with st.expander(
                "View Input and Prediction Data",
                expanded=True
            ):
                st.dataframe(
                    prediction_results,
                    use_container_width=True,
                    hide_index=True
                )

            csv = prediction_results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Prediction (CSV)",
                data=csv,
                file_name=(
                    f"bike_demand_prediction_"
                    f"{city_name}_{month}_{hour}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="individual_manual_download"
            )


    # STEP 2: BATCH PREDICTION
    with st.container(border=True):
        st.subheader("2. Batch Bike Demand Prediction")

        st.caption(
            "Generate multiple bike-sharing demand predictions "
            "by uploading your own input entries."
        )

        # Prepare CSV template
        template_columns = [
            "city_name",
            "weather_group",
            "working_day",
            "school_break",
            "month",
            "hour",
            "temp",
            "humidity",
            "wind_speed",
            "rain_1h",
            "snow_1h"
        ]

        template_data = [
            ["Berlin", "Clear", 1, 0, 5, 8, 18.0, 55, 3.2, 0.0, 0.0],
            ["Hamburg", "Rain", 1, 0, 10, 17, 12.0, 85, 5.5, 1.2, 0.0],
            ["Munich", "Clouds", 0, 1, 12, 14, 4.0, 70, 2.0, 0.0, 0.0]
        ]

        template_df = pd.DataFrame(
            template_data,
            columns=template_columns
        )

        template_csv = template_df.to_csv(
            index=False
        ).encode("utf-8")

        # Download and Upload side by side
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            with st.container(border=True, height=210):
                st.markdown("**Download CSV Template**")

                st.caption(
                    "Download the CSV template with example entries."
                )

                st.download_button(
                    label="Download CSV Template",
                    data=template_csv,
                    file_name="bike_demand_input_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="batch_template_download"
                )

        with col2:
            with st.container(border=True, height=210):
                st.markdown("**Upload Input Data**")

                st.caption(
                    "Upload your completed CSV file."
                )

                uploaded_file = st.file_uploader(
                    "Choose your CSV file",
                    type=["csv"],
                    key="batch_prediction_upload",
                    label_visibility="collapsed"
                )

        st.info(
            "You can modify the example entries or add new entries "
            "before uploading your CSV file."
        )

        # Read uploaded CSV
        if uploaded_file is not None:

            try:
                try:
                    batch_data = pd.read_csv(
                        uploaded_file,
                        encoding="utf-8-sig"
                    )

                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    batch_data = pd.read_csv(
                        uploaded_file,
                        encoding="cp1252"
                    )

            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
                batch_data = None

            except Exception as e:
                st.error(f"Could not read the uploaded CSV file: {e}")
                batch_data = None

            if batch_data is not None:

                if batch_data.empty:
                    st.warning(
                        "The uploaded CSV contains no prediction entries."
                    )

                else:
                    st.success(
                        f"CSV file uploaded successfully! "
                        f"{len(batch_data)} entries found."
                    )

                    st.divider()

                    if st.button(
                        "Generate Predictions",
                        type="primary",
                        use_container_width=True,
                        key="batch_predict"
                    ):

                        required_columns = [
                            "city_name",
                            "weather_group",
                            "working_day",
                            "school_break",
                            "month",
                            "hour",
                            "temp",
                            "humidity",
                            "wind_speed",
                            "rain_1h",
                            "snow_1h"
                        ]

                        missing_columns = [
                            col for col in required_columns
                            if col not in batch_data.columns
                        ]

                        if missing_columns:
                            st.error(
                                "The uploaded CSV is missing "
                                "the following columns: "
                                + ", ".join(missing_columns)
                            )

                        else:
                            try:
                                X_batch = batch_data[
                                    required_columns
                                ].copy()

                                predictions = model.predict(X_batch)

                                results = batch_data.copy()

                                results["predicted_bike_demand"] = [
                                    math.ceil(max(0, value))
                                    for value in predictions
                                ]

                                st.success(
                                    f"All {len(results)} predictions "
                                    "completed successfully!"
                                )

                                st.markdown("**Prediction Results**")

                                st.dataframe(
                                    results,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=200
                                )

                                results_csv = results.to_csv(
                                    index=False
                                ).encode("utf-8")

                                st.download_button(
                                    label="Download Prediction (CSV)",
                                    data=results_csv,
                                    file_name="bike_demand_prediction_results.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="batch_results_download"
                                )

                            except Exception as e:
                                st.error(f"Prediction failed: {e}")

with tab5:
    st.markdown(
        "<h1 style='text-align: center;'>"
        "Thanks for Exploring BikeCast! 🚲"
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align: center;'>"
        "Want to try it yourself?"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: center;'>"
        "Scan the QR code to visit our GitHub repository, "
        "where you can explore the project and find the link to try the BikeCast app."
        "</p>",
        unsafe_allow_html=True
    )

    # Center QR Code
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        with open("bikecast_qr.png", "rb") as image_file:
            qr_base64 = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; width: 100%;">
                <img src="data:image/png;base64,{qr_base64}"
                    width="250"
                    style="display: block; margin: auto;">
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Thank you for your time!"
        "</p>",
        unsafe_allow_html=True
    )
