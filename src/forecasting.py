import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =====================================================
# 1.Define Targets
# =====================================================

# Function 1: Define Account Ownership Target
def define_account_ownership_target(year, target_rate, description=None):
    """
    Define account ownership rate target.

    Parameters:
    -----------
    year : int
        Target year.
        
    target_rate : float
        Target percentage of adults owning a financial account.
        
    description : str
        Additional information about the target.

    Returns:
    --------
    dict
        Account ownership target information.
    """

    target = {
        "indicator": "Account Ownership Rate",
        "category": "Access",
        "year": year,
        "target_percentage": target_rate,
        "description": description
    }

    return target


# Function 2: Define Digital Payment Usage Target
def define_digital_payment_target(year, target_rate, description=None):
    """
    Define digital payment usage target.

    Parameters:
    -----------
    year : int
        Target year.

    target_rate : float
        Percentage of adults using digital payments.

    description : str
        Additional information about the target.

    Returns:
    --------
    dict
        Digital payment target information.
    """

    target = {
        "indicator": "Digital Payment Usage",
        "category": "Usage",
        "year": year,
        "target_percentage": target_rate,
        "description": description
    }

    return target


# Function 3: Combine All Targets
def create_financial_inclusion_targets(account_target,payment_target):
    """
    Combine multiple financial inclusion targets
    into a single dataframe.

    Parameters:
    -----------
    account_target : dict
        Account ownership target.

    payment_target : dict
        Digital payment target.

    Returns:
    --------
    pandas.DataFrame
        Target table.
    """

    targets = [account_target,payment_target]

    target_df = pd.DataFrame(targets)

    return target_df


# =====================================================
# 2. Select Approach Given sparse data 
# =====================================================

# Function 1: Prepare Findex Trend Data
def prepare_trend_data(
        df,
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Prepare time-series data for regression.

    Converts datetime into numeric year.
    """

    print("\n========== PREPARING TREND DATA ==========")


    trend_df = df[
        [
            date_column,
            value_column
        ]
    ].copy()



    # Convert date column

    trend_df[date_column] = pd.to_datetime(
        trend_df[date_column]
    )



    # Extract year

    trend_df["year"] = (
        trend_df[date_column]
        .dt.year
    )



    # Remove missing values

    trend_df = trend_df.dropna()



    # Sort

    trend_df = (
        trend_df
        .sort_values("year")
        .reset_index(drop=True)
    )


    print(trend_df)


    return trend_df[
        [
            "year",
            value_column
        ]
    ]


# Function 2: Fit Linear Trend Regression
def train_linear_trend_model(
        trend_df,
        value_column="value_numeric"):
    """
    Train linear trend:

    indicator = slope * year + intercept
    """

    print("\n========== TRAINING LINEAR MODEL ==========")



    X = trend_df[
        ["year"]
    ]


    y = trend_df[
        value_column
    ]



    model = LinearRegression()



    model.fit(
        X,
        y
    )



    prediction = model.predict(
        X
    )



    print(
        "Annual growth rate:",
        round(model.coef_[0],2)
    )


    print(
        "R2:",
        round(
            r2_score(
                y,
                prediction
            ),
            3
        )
    )


    return model


# Function 3: Fit Log Trend Regression
def train_log_trend_model(trend_df,year_column="year",value_column="value_numeric"):
    """
    Fit logarithmic growth:
    Indicator = a * log(year) + b
    Useful when growth slows over time.
    """

    print("\n========== LOG TREND MODEL ==========")


    X = np.log(trend_df[[year_column]])

    y = (trend_df[value_column])

    model = LinearRegression()

    model.fit(X, y)

    return model


# Function 4: Forecast Future Values
def forecast_future(
        model,
        future_years):
    """
    Predict future indicator values.
    """

    future_df = pd.DataFrame({

        "year": future_years

    })


    future_df["prediction"] = (
        model.predict(
            future_df
        )
    )


    return future_df


# Function 5: Visualize Trend
def plot_trend_forecast(historical_df,forecast_df,year_column="year",value_column="value_numeric"):

    """
    Plot historical and forecast trend.
    """

    plt.figure(figsize=(10,5))

    plt.plot(historical_df[year_column],historical_df[value_column],
        marker="o",label="Observed")

    plt.plot(forecast_df["year"],forecast_df["prediction"],
        marker="o",linestyle="--",label="Forecast")

    plt.title( "Financial Inclusion Trend Forecast" )

    plt.xlabel("Year")

    plt.ylabel("Indicator Value (%)")

    plt.grid(True)

    plt.legend()

    plt.show()


# Function 3: Event-Augmented Model
def build_event_augmented_model(indicator_df,event_features,year_col="year",
        value_col="value_numeric"):
    """
    Model:
    Indicator =Trend + Event Effects

    event_features example:

    year | telebirr | mpesa | safaricom

    2021 |    1     |   0   |     0
    2023 |    1     |   1   |     1

    """


    print("\n===== EVENT AUGMENTED MODEL =====")

    # Merge trend and events
    df = indicator_df.merge(event_features, on=year_col,how="left")

    df = df.fillna(0)

    features = [ year_col ] + [col for col in event_features.columns
                            if col != year_col]

    X = df[features]

    y = df[value_col]

    model = LinearRegression()

    model.fit( X, y )

    print( "Event coefficients:" )

    coefficients = pd.DataFrame({"feature":features,"impact":model.coef_})

    print(coefficients)

    return model, coefficients


# Function : Scenario Analysis
def scenario_analysis(model,future_data,scenarios):
    """
    Test future assumptions.

    Example:

    Optimistic:
    Strong event effects

    Baseline:
    Current trend

    Conservative:
    Slow adoption

    """


    print( "\n===== SCENARIO ANALYSIS =====")

    results = {}

    for scenario, adjustment in scenarios.items():

        prediction = (model.predict(future_data) + adjustment)

        results[scenario] = prediction

        print(scenario, prediction)

    return pd.DataFrame(results)



# Function: Plot Forecast Scenarios
def plot_scenarios(years,scenario_results):

    plt.figure(figsize=(10,5))

    for column in scenario_results.columns:

        plt.plot(years,scenario_results[column],marker="o",label=column)

    plt.title("Financial Inclusion Scenario Forecast")

    plt.xlabel( "Year" )

    plt.ylabel("Indicator (%)")

    plt.grid(True)

    plt.legend()

    plt.show()

def train_model_with_uncertainty(
        historical_df,
        year_col="year",
        value_col="value_numeric"):
    """
    Train linear trend model and calculate
    historical prediction uncertainty.
    """


    print("\n========== TRAINING MODEL ==========")


    X = historical_df[[year_col]]

    y = historical_df[value_col]



    model = LinearRegression()


    model.fit(
        X,
        y
    )


    # Historical prediction

    y_pred = model.predict(
        X
    )



    # Calculate error

    rmse = np.sqrt(
        mean_squared_error(
            y,
            y_pred
        )
    )


    print(
        "RMSE:",
        round(rmse,2)
    )


    return model, rmse




# =====================================================
# Function 2: Generate Forecast Confidence Interval
# =====================================================
def generate_confidence_interval(
        model,
        rmse,
        future_years,
        confidence_level=0.95):
    """
    Generate forecast with uncertainty range.

    Forecast:
        predicted value

    Lower:
        prediction - uncertainty

    Upper:
        prediction + uncertainty
    """


    print(
        "\n========== CONFIDENCE INTERVAL =========="
    )


    future = pd.DataFrame({

        "year":future_years

    })



    predictions = model.predict(
        future
    )


    # Approximate 95% confidence range

    z_score = 1.96



    uncertainty = (
        z_score
        *
        rmse
    )



    result = pd.DataFrame({

        "year":future_years,

        "forecast":predictions,

        "lower_bound":
        predictions - uncertainty,

        "upper_bound":
        predictions + uncertainty

    })


    print(result)


    return result




# =====================================================
# Function 3: Generate Scenario Ranges
# =====================================================
def create_scenario_range(
        forecast_df,
        event_effect_col="event_forecast"):
    """
    Create future uncertainty scenarios.

    Optimistic:
        stronger adoption

    Base:
        expected path

    Pessimistic:
        slower adoption
    """


    print(
        "\n========== SCENARIO RANGE =========="
    )


    scenarios = forecast_df.copy()



    scenarios["optimistic"] = (
        scenarios[event_effect_col]
        +
        5
    )


    scenarios["base"] = (
        scenarios[event_effect_col]
    )


    scenarios["pessimistic"] = (
        scenarios[event_effect_col]
        -
        5
    )



    return scenarios




# =====================================================
# Function 4: Document Model Limitations
# =====================================================
def document_limitations():
    """
    Explicitly record forecasting limitations.
    """


    print(
        "\n========== MODEL LIMITATIONS =========="
    )


    limitations = [

        "Only five Findex observations are available.",

        "Forecast assumes historical relationships continue.",

        "Event impacts are estimated, not experimentally proven.",

        "External shocks (economic, political, regulatory) are not fully captured.",

        "Mobile money registration may not represent active usage.",

        "Regional and demographic differences may affect national trends.",

        "Future policy and technology changes introduce uncertainty."

    ]



    for limitation in limitations:

        print("-", limitation)



    return limitations




# =====================================================
# Function 5: Plot Forecast Uncertainty
# =====================================================
def plot_uncertainty(
        uncertainty_df):


    plt.figure(figsize=(10,5))


    plt.plot(
        uncertainty_df["year"],
        uncertainty_df["forecast"],
        marker="o",
        label="Forecast"
    )


    plt.fill_between(
        uncertainty_df["year"],
        uncertainty_df["lower_bound"],
        uncertainty_df["upper_bound"],
        alpha=0.3,
        label="95% Confidence Interval"
    )



    plt.title(
        "Financial Inclusion Forecast Uncertainty"
    )


    plt.xlabel(
        "Year"
    )


    plt.ylabel(
        "Indicator (%)"
    )


    plt.grid(True)

    plt.legend()

    plt.show()