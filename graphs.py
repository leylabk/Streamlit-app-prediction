import streamlit as st
import plotly.express as px
import pandas as pd

# City colors
city_colors = {
    "Berlin": "#2878B5",
    "Dresden": "#E58B35",
    "Köln": "#329B75"
}

# Hourly Usage Patterns
def show_hourly_usage(df):

    with st.container(border=True):

        st.subheader("Hourly Usage Patterns")
        st.caption("Average bike usage throughout the day")

        hourly_data = (
            df.groupby(
                ["hour", "city_name"],
                observed=True
            )["total_bike_usage"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly_data,
            x="hour",
            y="total_bike_usage",
            color="city_name",
            markers=True,
            color_discrete_map=city_colors
        )

        fig.update_traces(
            line=dict(width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{y:,.0f}</b><extra></extra>"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Hour of the Day",
            yaxis_title="Average Bike Usage",
            hovermode="x",
            height=400
        )

        fig.update_xaxes(
            tickmode="linear",
            tick0=0,
            dtick=2,
            range=[0, 23],
            showgrid=False
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#EAECEF",
            zeroline=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False
            }
        )


# Monthly Trends
def show_monthly_trends(df):

    with st.container(border=True):

        st.subheader("Monthly Trends")
        st.caption("Average bike usage across the year (August–July)")

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
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "select2d",
                    "lasso2d"
                ]
            }
        )


# Temperature and Bike Usage
def show_temperature_usage(df):

    with st.container(border=True):

        st.subheader("Temperature and Bike Usage")
        st.caption("Average bike usage across temperature ranges by city")

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
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "select2d",
                    "lasso2d"
                ]
            }
        )



# City Comparison
def show_city_comparison(df):

    with st.container(border=True):

        st.subheader("City Comparison")
        st.caption("Average hourly bike usage by city")

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
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "select2d",
                    "lasso2d"
                ]
            }
        )



# Temperature, Weather Condition and Bike Usage
# Weather colors
Weather_colors = {
    "Clear": "#FF9800",
    "Clouds": "#78909C",
    "Rain": "#3949AB",
    "Drizzle": "#03A9F4",
    "Snow": "#00BCD4",
    "Mist": "#E2EAF0",
    "Fog": "#B8CBD8"
}

# Weather symbols
Weather_emojis = {
    "Clear": "☀",
    "Clouds": "☁",
    "Rain": "🌧️",
    "Drizzle": "☂",
    "Snow": "❄",
    "Mist": "~",
    "Fog": "≋"
}


def show_weather_usage(df):

    with st.container(border=True):

        st.subheader("Temperature, Weather Condition and Bike Usage")
        st.caption("Average bike usage by city and weather condition")

        weather_city_usage = (
            df.groupby(
                ["city_name", "weather_group"],
                observed=True
            )["total_bike_usage"]
            .mean()
            .reset_index()
        )

        # Add weather symbols
        weather_city_usage["weather_symbol"] = (
            weather_city_usage["weather_group"]
            .map(Weather_emojis)
        )

        # Create grouped bar chart
        fig = px.bar(
            weather_city_usage,
            x="city_name",
            y="total_bike_usage",
            color="weather_group",
            text="weather_symbol",
            barmode="group",
            color_discrete_map=Weather_colors,
            category_orders={
                "weather_group": list(Weather_colors.keys())
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
            trace.textfont.color = Weather_colors.get(
                trace.name, "#455A64"
            )
            trace.textfont.size = 15 if trace.name == "Rain" else 20

        # Calculate maximum usage
        max_usage = weather_city_usage["total_bike_usage"].max()

        # Layout styling
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
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "select2d",
                    "lasso2d"
                ]
            }
        )
