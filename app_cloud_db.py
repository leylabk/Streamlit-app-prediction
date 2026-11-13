
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
                                  "Prediction Tool", 
                                  "Prediction Tool2"])

with tab1:
    st.write("""
        ### About the Project
        This project aims to analyze and predict hourly bike-sharing demand based on various temporal, weather-related, and calendar features. 
        By exploring historical bike usage data, we investigate how different factors influence bicycle rental patterns across cities.

        ### About the Dataset
        This project combines bike-sharing, weather, and calendar data
        from different sources to analyze and predict hourly bicycle usage.
        This project combines bike-sharing, weather, and calendar data
        from different sources to analyze and predict hourly bicycle usage.

        **Data Sources:**

        - **Bike-Sharing Data:** Historical bike-sharing records obtained
        from the [European Bike Sharing Dataset](https://github.com/tumftm/european-bike-sharing-dataset).
        - **Weather Data:** Weather information collected using the
        [OpenWeather API](https://openweathermap.org/api), including temperature, humidity, rainfall,
        and snowfall.
        - **Calendar Data:** Working day and school holiday information
        generated using Python libraries.

        ## Features Used in the Analysis
        - **Temporal Features:** Year, month, day, and hour.
        - **Weather Features:** Temperature, humidity, rainfall, and snowfall.
        - **Calendar Features:** School breaks and working days.
        - **Location Feature:** City name.
        - **Weather Categories:** Grouped weather conditions.

        **Target Variable:** `total_bike_usage`

        Represents the total number of bicycles used within each hour.
        ### What Can You Explore?

        - **Data Exploration:** Examine the dataset and identify usage patterns.
        - **Data Visualization:** Investigate how weather, time, and calendar
        factors affect bicycle demand.
        - **Model Insights:** Explore the performance and important features
        of the selected CatBoost model.
        - **Demand Prediction:** Estimate hourly bicycle usage based on
        selected conditions.

        ### Explore the App!
        Navigate through the pages to explore bike-sharing patterns, 
        gain insights into the factors influencing bicycle usage, 
        and predict hourly bike-sharing demand using our best-performing CatBoost model.
    """)

    st.caption(
    "Dataset source: Waldner et al. (2025), "
    "Data-Driven Insights into (E-)Bike-Sharing, Transportation. "
    "Licensed under CC BY-NC 4.0."
    )


with tab2:
    st.title("Data Visualization")
    st.markdown(
        "Explore bike-sharing patterns across different "
        "cities, hours, months, and temperature conditions."
    )

    st.divider()

    # Consistent city colors
    city_colors = {
        "Berlin": "#2878B5",
        "Dresden": "#E58B35",
        "Köln": "#329B75"
    }

    # First row
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Hourly Usage Patterns")
            st.caption("Average bike usage throughout the day")

            # Calculate average usage by hour and city
            hourly_data = (
                df.groupby(["hour", "city_name"], observed=True)["total_bike_usage"]
                .mean()
                .reset_index()
            )
            import plotly.express as px
            # Create chart
            fig = px.line(
                hourly_data,
                x="hour",
                y="total_bike_usage",
                color="city_name",
                markers=True,
                color_discrete_map=city_colors
            )

            # Simple line styling
            fig.update_traces(
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{y:,.0f}</b><extra></extra>"
            )

            # Clean layout
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Hour of the Day",
                yaxis_title="Average Bike Usage",
                hovermode="x",

                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=11)
                ),

                margin=dict(l=20, r=20, t=55, b=30),
                height=400,

                plot_bgcolor="white",
                paper_bgcolor="white",

                font=dict(
                    family="Arial",
                    size=12,
                    color="#455A64"
                )
            )

            # X-axis
            fig.update_xaxes(
                tickmode="linear",
                tick0=0,
                dtick=2,
                range=[0, 23],
                showgrid=False,
                showline=True,
                linecolor="#D9DEE3",
                tickfont=dict(size=11)
            )

            # Y-axis
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#EAECEF",
                zeroline=False,
                showline=False,
                tickfont=dict(size=11)
            )

            # Display chart
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "toImage",
                        "select2d",
                        "lasso2d"
                    ]
                }
            )

    with col2:

        with st.container(border=True):
            st.subheader("Monthly Trends")
            st.caption("Average bike usage across the year (August–July)")

            import plotly.express as px

            # Define custom month order
            month_order = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]

            month_labels = [
                "Aug", "Sep", "Oct", "Nov", "Dec",
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"
            ]

            month_map = dict(zip(month_order, month_labels))

            # Prepare monthly data
            monthly_df = df.copy()

            monthly_df["month_label"] = monthly_df["month"].map(month_map)

            monthly_data = (
                monthly_df.groupby(
                    ["month_label", "city_name"],
                    observed=True
                )["total_bike_usage"]
                .mean()
                .reset_index()
            )

            # Preserve August-July order
            monthly_data["month_label"] = pd.Categorical(
                monthly_data["month_label"],
                categories=month_labels,
                ordered=True
            )

            monthly_data = monthly_data.sort_values("month_label")

            # Create interactive chart
            fig = px.line(
                monthly_data,
                x="month_label",
                y="total_bike_usage",
                color="city_name",
                markers=True,
                color_discrete_map=city_colors,
                category_orders={"month_label": month_labels}
            )

            # Line and marker styling
            fig.update_traces(
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{y:,.0f}</b><extra></extra>"
            )

            # Layout styling
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Month",
                yaxis_title="Average Bike Usage",
                hovermode="x",
                clickmode="event",

                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=11)
                ),

                margin=dict(l=20, r=20, t=55, b=30),
                height=400,

                plot_bgcolor="white",
                paper_bgcolor="white",

                font=dict(
                    family="Arial",
                    size=12,
                    color="#455A64"
                )
            )

            # X-axis
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=month_labels,
                showgrid=False,
                showline=True,
                linecolor="#D9DEE3",
                tickfont=dict(size=11)
            )

            # Y-axis
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#EAECEF",
                zeroline=False,
                showline=False,
                tickfont=dict(size=11)
            )

            # Display chart
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "toImage",
                        "select2d",
                        "lasso2d"
                    ]
                }
            )

 
    # Second row
    col3, col4 = st.columns(2, gap="large")

    with col3:
        with st.container(border=True):
            st.subheader("Temperature and Bike Usage")
            st.caption("Average bike usage across temperature ranges by city")

            import plotly.express as px

            # Group temperature into 15 ranges
            df_temp = df.copy()

            df_temp["temp_group"] = pd.cut(
                df_temp["temp"],
                bins=15
            )

            # Calculate average usage
            temp_usage = (
                df_temp.groupby(
                    ["temp_group", "city_name"],
                    observed=True
                )["total_bike_usage"]
                .mean()
                .reset_index()
            )

            # Calculate temperature midpoint
            temp_usage["temp_mid"] = temp_usage["temp_group"].apply(
                lambda x: x.mid
            )

            # Create interactive chart
            fig = px.line(
                temp_usage,
                x="temp_mid",
                y="total_bike_usage",
                color="city_name",
                markers=True,
                color_discrete_map=city_colors
            )

            # Line and marker styling
            fig.update_traces(
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{y:,.0f}</b><extra></extra>"
            )

            # Layout styling
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Temperature",
                yaxis_title="Average Bike Usage",
                hovermode="x",

                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=11)
                ),

                margin=dict(l=20, r=20, t=55, b=30),
                height=400,

                plot_bgcolor="white",
                paper_bgcolor="white",

                font=dict(
                    family="Arial",
                    size=12,
                    color="#455A64"
                )
            )

            # X-axis
            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linecolor="#D9DEE3",
                tickfont=dict(size=11)
            )

            # Y-axis
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#EAECEF",
                zeroline=False,
                showline=False,
                tickfont=dict(size=11)
            )

            # Display chart
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "toImage",
                        "select2d",
                        "lasso2d"
                    ]
                }
            )

    with col4:

        with st.container(border=True):
            st.subheader("City Comparison")
            st.caption("Average hourly bike usage by city")

            import plotly.express as px

            # Calculate average usage by city
            city_usage = (
                df.groupby("city_name")["total_bike_usage"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )

            # Create interactive bar chart
            fig = px.bar(
                city_usage,
                x="total_bike_usage",
                y="city_name",
                color="city_name",
                color_discrete_map=city_colors,
                orientation="h"
            )

            # Bar styling
            fig.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{x:,.0f}</b><extra></extra>"
            )

            # Layout styling
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Average Bike Usage",
                yaxis_title=None,
                hovermode="closest",
                showlegend=False,

                margin=dict(l=20, r=20, t=25, b=35),
                height=400,

                plot_bgcolor="white",
                paper_bgcolor="white",

                font=dict(
                    family="Arial",
                    size=12,
                    color="#455A64"
                )
            )

            # X-axis
            fig.update_xaxes(
                showgrid=True,
                gridcolor="#EAECEF",
                zeroline=False,
                showline=False,
                tickfont=dict(size=11)
            )

            # Y-axis
            fig.update_yaxes(
                showgrid=False,
                showline=False,
                tickfont=dict(size=12),
                categoryorder="array",
                categoryarray=city_usage["city_name"].tolist()
            )

            # Display chart
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "toImage",
                        "select2d",
                        "lasso2d"
                    ]
                }
            )

# Third row
    col5, col6 = st.columns(2, gap="large")

    with st.container(border=True):
        st.subheader("Temperature, Weather Condition and Bike Usage")
        st.caption("Average bike usage by city and weather condition")

        import plotly.express as px

        weather_city_usage = (
            df.groupby(
                ["city_name", "weather_group"],
                observed=True
            )["total_bike_usage"]
            .mean()
            .reset_index()
        )

        # Weather colors
        weather_colors = {
            "Clear": "#FF9800",
            "Clouds": "#78909C",
            "Rain":"#3949AB",
            "Drizzle": "#03A9F4",
            "Snow": "#00BCD4",
            "Mist": "#E2EAF0",
            "Fog":"#B8CBD8"
        }

        # Weather symbols
        weather_emojis = {
            "Clear": "☀",
            "Clouds": "☁",
            "Rain": "🌧️",
            "Drizzle": "☂",
            "Snow": "❄",
            "Mist": "~",
            "Fog": "≋"
        }

        # Add symbols above bars
        weather_city_usage["weather_symbol"] = (
            weather_city_usage["weather_group"]
            .map(weather_emojis)
        )

        # Create grouped bar chart
        fig = px.bar(
            weather_city_usage,
            x="city_name",
            y="total_bike_usage",
            color="weather_group",
            text="weather_symbol",
            barmode="group",
            color_discrete_map=weather_colors,
            category_orders={
                "weather_group": list(weather_colors.keys())
            }
        )

        # Bar and symbol styling
        fig.update_traces(
            marker_line_width=0,
            textposition="outside",
            texttemplate="%{text}",
            textfont=dict(size=20),
            cliponaxis=False,
            hovertemplate="<b>%{y:,.0f}</b><extra></extra>"
        )

        # Match symbol colors with bar colors
        for trace in fig.data:
            trace.textfont.color = weather_colors.get(
                trace.name, "#455A64"
            )
            trace.textfont.size = 15 if trace.name == "Rain" else 20
        # Layout styling
        max_usage = weather_city_usage["total_bike_usage"].max()

        fig.update_layout(
            template="plotly_white",
            xaxis_title=None,
            yaxis_title="Average Bike Usage",
            hovermode="closest",

            legend=dict(
                title="Weather Condition",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=11)
            ),

            bargap=0.2,
            bargroupgap=0.05,

            margin=dict(l=20, r=20, t=65, b=35),
            height=480,

            plot_bgcolor="white",
            paper_bgcolor="white",

            font=dict(
                family="Arial",
                size=12,
                color="#455A64"
            )
        )

        # X-axis
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            linecolor="#D9DEE3",
            tickfont=dict(size=11)
        )

        # Y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#EAECEF",
            zeroline=False,
            showline=False,
            tickfont=dict(size=11),
            range=[0, max_usage * 1.2]
        )

        # Display chart
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "select2d",
                    "lasso2d"
                ]
            }
        )
  



with tab3:
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

        prediction = model.predict(input_data)

        st.success("Prediction completed!")

        st.metric(
            label="Predicted value for hourly bicycle usage",
            value=f"{prediction[0]:,.2f}"
        )

        with st.expander("View Input Data"):
            st.dataframe(input_data)

with tab4:

    st.header("Automatic Bike Demand Prediction")

    st.caption(
        "Select a city, date, and time. Weather and calendar "
        "information will be retrieved automatically."
    )

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