def select_forecast_model(
        model_name,
        forecast_df):


    if model_name=="Linear Trend":

        return forecast_df[
            forecast_df.model=="linear"
        ]


    elif model_name=="ARIMA":

        return forecast_df[
            forecast_df.model=="arima"
        ]


    elif model_name=="Prophet":

        return forecast_df[
            forecast_df.model=="prophet"
        ]


    return forecast_df