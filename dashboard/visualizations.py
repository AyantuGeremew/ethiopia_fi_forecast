import plotly.express as px
import plotly.graph_objects as go



def time_series_plot(df,column):

    fig = px.line(
        df,
        x="date",
        y=column,
        markers=True,
        title=f"{column} Trend"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=column
    )

    return fig



def channel_comparison(df):

    fig = px.line(
        df,
        x="date",
        y=[
            "mobile_money",
            "bank_account",
            "atm"
        ],
        title="Channel Comparison"
    )

    return fig



def forecast_plot(df):

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["forecast"],
            name="Forecast"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["upper_ci"],
            line=dict(dash="dash"),
            name="Upper Confidence"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["lower_ci"],
            line=dict(dash="dash"),
            name="Lower Confidence"
        )
    )


    fig.update_layout(
        title="Forecast with Confidence Interval"
    )


    return fig



def target_progress_chart(current,target):


    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current,
            title={
                "text":
                f"Progress Toward {target}% Target"
            },
            gauge={
                "axis":{
                    "range":[0,target]
                }
            }
        )
    )


    return fig