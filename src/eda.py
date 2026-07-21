import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1. Data Overview
# ==========================================================

#  Summarize Dataset
def summarize_dataset(df):
    """
    Summarize dataset by:
    - record_type
    - pillar
    - source_type
    """

    print("\n========== DATASET SUMMARY ==========")

    columns = ["record_type", "pillar", "source_type"]

    for column in columns:

        if column not in df.columns:
            print(f"\nColumn '{column}' not found.")
            continue

        print(f"\nSummary by {column}")

        summary = (df[column].value_counts(dropna=False).reset_index())

        summary.columns = [column, "Count"]

        print(summary)


# Temporal Coverage Visualization
def plot_temporal_coverage(df,date_column="observation_date",indicator_column="indicator_code"):
    """
    Display which years contain data
    for each indicator.
    """
    print("\n========== TEMPORAL COVERAGE ==========")

    if date_column not in df.columns:
        print(f"{date_column} column not found.")
        return

    df = df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    df["Year"] = df[date_column].dt.year

    coverage = (df.groupby(["Year", indicator_column]).size().unstack(fill_value=0))

    print(coverage)

    coverage.plot(figsize=(12,6),marker="o")

    plt.title("Temporal Coverage by Indicator")

    plt.xlabel("Year")

    plt.ylabel("Number of Records")

    plt.grid(True)

    plt.tight_layout()

    plt.show()



#  Assess Data Quality
def assess_confidence_levels(df, confidence_column="confidence"):
    """
    Assess data quality by examining
    confidence level distribution.
    """

    print("\n========== CONFIDENCE DISTRIBUTION ==========")

    if confidence_column not in df.columns:
        print(f"{confidence_column} column not found.")
        return

    confidence = (df[confidence_column].value_counts(dropna=False))

    print(confidence)

    confidence.plot(kind="bar",figsize=(6,4))

    plt.title("Confidence Level Distribution")

    plt.xlabel("Confidence")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()



#  Identify Sparse Indicators
def identify_sparse_indicators(df,indicator_column="indicator_code",threshold=10):
    """
    Identify indicators having fewer than
    threshold records.
    """

    print("\n========== SPARSE INDICATORS ==========")

    counts = (df[indicator_column].value_counts().reset_index())

    counts.columns = ["Indicator","Record Count"]

    sparse = counts[counts["Record Count"] < threshold]

    print(sparse)

    return sparse

# ==========================================================
# 2. Access Analysis
# ==========================================================

# Plot Account Ownership Trend (2011-2024)
def plot_account_ownership_trajectory(
        df,
        indicator_code="ACCOUNT_OWNERSHIP",
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Plot Ethiopia's account ownership trajectory
    from 2011 to 2024.

    Parameters
    ----------
    df : pandas.DataFrame
        Financial inclusion dataset

    indicator_code : str
        Indicator representing account ownership

    date_column : str
        Date of observation

    value_column : str
        Measured ownership percentage

    Returns
    -------
    DataFrame
        Yearly account ownership trend
    """


    print("\n========== ACCOUNT OWNERSHIP TRAJECTORY ==========")


    # -----------------------------------------
    # 1. Filter account ownership indicator
    # -----------------------------------------

    ownership = df[
        df["indicator_code"] == indicator_code
    ].copy()


    if ownership.empty:
        print(
            "No account ownership data found."
        )
        return None



    # -----------------------------------------
    # 2. Convert date and extract year
    # -----------------------------------------

    ownership[date_column] = pd.to_datetime(
        ownership[date_column]
    )


    ownership["year"] = (
        ownership[date_column]
        .dt.year
    )



    # -----------------------------------------
    # 3. Calculate yearly ownership
    # -----------------------------------------

    yearly_trend = (
        ownership
        .groupby("year")[value_column]
        .mean()
        .reset_index()
    )



    # Keep only 2011-2024

    yearly_trend = yearly_trend[
        yearly_trend["year"]
        .between(2011, 2024)
    ]



    # -----------------------------------------
    # 4. Plot trajectory
    # -----------------------------------------

    plt.figure(figsize=(10,5))


    plt.plot(
        yearly_trend["year"],
        yearly_trend[value_column],
        marker="o"
    )


    plt.title(
        "Ethiopia Account Ownership Trajectory (2011-2024)"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Account Ownership (%)"
    )


    plt.xticks(
        yearly_trend["year"],
        rotation=45
    )


    plt.grid(True)

    plt.tight_layout()

    plt.show()

    return yearly_trend

# Calculate and Visualize Growth Rates
def calculate_account_growth_rates(trend_df, year_column="year",value_column="value_numeric"):
    """
    Calculate account ownership growth between survey years
    and visualize the growth rate.

    Parameters
    ----------
    trend_df : DataFrame
        Account ownership trend data

    year_column : str
        Year column

    value_column : str
        Account ownership percentage column

    Returns
    -------
    DataFrame
        Growth rate summary
    """

    print("\n========== ACCOUNT OWNERSHIP GROWTH ==========")

    # Sort by survey year
    growth_df = (trend_df.sort_values(year_column).copy())

    # Calculate growth
    growth_df["previous_value"] = (growth_df[value_column].shift(1))

    growth_df["previous_year"] = (growth_df[year_column].shift(1))

    # Percentage point change
    growth_df["growth_pp"] = (growth_df[value_column]-growth_df["previous_value"])

    # Years between surveys
    growth_df["year_difference"] = (growth_df[year_column]-growth_df["previous_year"])

    # Annual growth rate
    growth_df["annual_growth_rate"] = (growth_df["growth_pp"]/growth_df["year_difference"])

    print(
        growth_df[
            [
                year_column,
                value_column,
                "growth_pp",
                "annual_growth_rate"
            ]
        ]
    )

    # Visualize growth
    plot_data = growth_df.dropna()

    plt.figure(figsize=(10,5))

    plt.bar(plot_data[year_column].astype(str),plot_data["growth_pp"])

    plt.axhline(y=0)

    plt.title("Ethiopia Account Ownership Growth Between Survey Years")

    plt.xlabel("Survey Year")

    plt.ylabel("Growth (Percentage Points)")

    plt.xticks( rotation=45)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.show()

    return growth_df

# Analyze Account Ownership Slowdown (2021-2024)
def analyze_2021_2024_slowdown(
        df,
        ownership_indicator="ACCOUNT_OWNERSHIP",
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Analyze account ownership growth slowdown
    between 2021 and 2024.
    """

    print("\n========== ACCOUNT OWNERSHIP SLOWDOWN ==========")

    # Filter account ownership data
    ownership = df[df["indicator_code"] == ownership_indicator].copy()

    if ownership.empty:
        print("No account ownership data found.")
        return None

    # Convert date
    ownership[date_column] = pd.to_datetime(ownership[date_column])

    ownership["year"] = (ownership[date_column].dt.year)

    # Yearly ownership values
    yearly = (ownership.groupby("year")[value_column].mean().reset_index())

    # Calculate growth
    yearly["growth_pp"] = (yearly[value_column].diff())

    print("\nAccount Ownership Trend:")
    print(yearly)

    # Calculate 2021-2024 growth
    period = yearly[yearly["year"].between(2021, 2024)]

    slowdown_growth = (period[value_column].iloc[-1]-period[value_column].iloc[0])

    print("\n2021-2024 Growth:",round(slowdown_growth,2),"percentage points")

    # Visualization
    plt.figure(figsize=(8,5))
    plt.plot(yearly["year"],yearly[value_column],marker="o")

    plt.title("Ethiopia Account Ownership Growth (2011-2024)")

    plt.xlabel("Year")

    plt.ylabel("Account Ownership (%)")

    plt.grid(True)

    plt.show()

    return yearly

# Explain Possible Slowdown Factors
def explain_slowdown_factors():
    """
    Provide possible factors explaining why
    account ownership growth slowed despite
    mobile money expansion.
    """

    print("\n========== POSSIBLE EXPLANATIONS ==========")

    factors = {

        "Market Saturation":
        "Most easily reachable populations already have accounts; remaining groups are harder to reach.",

        "Rural Access Challenges":
        "Remote communities may still lack agents, infrastructure, and reliable connectivity.",

        "Gender Gap":
        "Women may face barriers related to income, digital literacy, and access.",

        "Mobile Money Conversion Gap":
        "Mobile money growth can increase transactions without creating new account owners.",

        "Digital Literacy":
        "Limited skills reduce adoption of digital financial services.",

        "Economic Conditions":
        "Inflation and income constraints may reduce demand for financial services.",

        "Infrastructure Limitations":
        "Internet coverage, electricity, and agent network limitations may slow adoption."
    }

    for factor, explanation in factors.items():

        print(f"\n{factor}:")
        print(explanation)

    return factors

def analyze_mobile_money_penetration(
        df,
        indicator_code="MOBILE_MONEY_ACCOUNT",
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Analyze Ethiopia mobile money account penetration
    trend from 2014 to 2024.

    Parameters
    ----------
    df : DataFrame
        Financial inclusion dataset

    indicator_code : str
        Mobile money indicator code

    date_column : str
        Observation date column

    value_column : str
        Penetration percentage value

    Returns
    -------
    DataFrame
        Mobile money trend with growth rates
    """

    print("\n========== MOBILE MONEY PENETRATION ==========")

    #  Filter mobile money records
    mobile_data = df[df["indicator_code"] == indicator_code].copy()

    if mobile_data.empty:
        print(
            "No mobile money data found."
        )
        return None

    #  Extract year
    mobile_data[date_column] = pd.to_datetime(mobile_data[date_column])

    mobile_data["year"] = (mobile_data[date_column].dt.year)

    # Keep 2014-2024
    mobile_data = mobile_data[mobile_data["year"].between(2014, 2024)]

    #  Calculate yearly penetration
    trend = (mobile_data.groupby("year")[value_column].mean().reset_index())

    # Calculate annual growth
    trend["growth_pp"] = (trend[value_column].diff())

    trend["growth_rate_%"] = (trend[value_column].pct_change()* 100)

    print("\nMobile Money Trend:")

    print(trend)

    # Visualize penetration trend
    plt.figure(figsize=(10,5))

    plt.plot(trend["year"],trend[value_column],marker="o")

    plt.title("Ethiopia Mobile Money Account Penetration (2014-2024)")

    plt.xlabel("Year")

    plt.ylabel("Mobile Money Account Penetration (%)")

    plt.grid(True)

    plt.xticks(trend["year"],rotation=45)

    plt.tight_layout()

    plt.show()

    # Visualize growth
    plt.figure(figsize=(10,5))

    plt.bar(trend["year"].astype(str),trend["growth_pp"])

    plt.title("Annual Mobile Money Penetration Growth")

    plt.xlabel("Year")

    plt.ylabel("Growth (Percentage Points)")

    plt.xticks(rotation=45)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.show()

    return trend


def analyze_digital_payment_adoption(
        df,
        indicator_code="DIGITAL_PAYMENT",
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Examine digital payment adoption patterns over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Financial inclusion dataset

    indicator_code : str
        Digital payment indicator

    date_column : str
        Observation date

    value_column : str
        Adoption percentage/value

    Returns
    -------
    DataFrame
        Digital payment adoption trend
    """

    print("\n========== DIGITAL PAYMENT ADOPTION ==========")

    # Filter digital payment records
    payment_data = df[df["indicator_code"] == indicator_code].copy()

    if payment_data.empty:

        print("No digital payment data found.")

        return None

    # Extract year
    payment_data[date_column] = pd.to_datetime(payment_data[date_column])

    payment_data["year"] = (payment_data[date_column].dt.year)

    # Calculate yearly adoption
    
    adoption_trend = (
        payment_data
        .groupby("year")[value_column]
        .mean()
        .reset_index()
    )

    # Calculate growth
    adoption_trend["growth_pp"] = (adoption_trend[value_column].diff())

    adoption_trend["growth_rate_%"] = (
        adoption_trend[value_column]
        .pct_change()
        * 100
    ).round(2)

    print("\nDigital Payment Adoption Trend:")

    print(adoption_trend)

    # Plot adoption trend
    plt.figure(figsize=(10,5))

    plt.plot(adoption_trend["year"],adoption_trend[value_column],marker="o")

    plt.title("Ethiopia Digital Payment Adoption Trend")

    plt.xlabel("Year")

    plt.ylabel("Digital Payment Adoption (%)")

    plt.grid(True)

    plt.xticks(adoption_trend["year"], rotation=45)

    plt.tight_layout()

    plt.show()

    # Plot growth pattern
    plt.figure(figsize=(10,5))

    plt.bar(adoption_trend["year"].astype(str), adoption_trend["growth_pp"])

    plt.title("Digital Payment Adoption Growth")

    plt.xlabel("Year")

    plt.ylabel("Growth (Percentage Points)")

    plt.xticks(rotation=45)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.show()

    return adoption_trend

def analyze_registered_active_gap(
        df,
        registered_indicator="REGISTERED_MOBILE_ACCOUNTS",
        active_indicator="ACTIVE_MOBILE_USERS",
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Compare registered mobile money accounts
    with active/survey-reported usage.

    Identifies the gap between account registration
    and actual usage.
    """

    print("\n========== REGISTERED VS ACTIVE GAP ==========")

    # Select required indicators
    data = df[
        df["indicator_code"]
        .isin(
            [
                registered_indicator,
                active_indicator
            ]
        )
    ].copy()


    if data.empty:

        print("Required indicators not found.")

        return None

    # Convert date
    data[date_column] = pd.to_datetime(data[date_column])

    data["year"] = (data[date_column].dt.year)

    # Aggregate yearly values
    comparison = (
        data
        .groupby(
            [
                "year",
                "indicator_code"
            ]
        )[value_column]
        .mean()
        .reset_index()
    )

    print(
        comparison
    )

    # Pivot for comparison
    pivot = comparison.pivot(
        index="year",
        columns="indicator_code",
        values=value_column
    )

    # Calculate gap
    if (registered_indicator in pivot.columns and active_indicator in pivot.columns):

        pivot["usage_gap"] = (pivot[registered_indicator] - pivot[active_indicator])

        print( "\nRegistered-Active Gap:")

        print(pivot)

    # Visualization
    pivot[
        [
            registered_indicator,
            active_indicator
        ]
    ].plot(
        figsize=(10,5),
        marker="o"
    )

    plt.title("Registered Accounts vs Active Usage")

    plt.ylabel("Percentage / Number of Accounts")

    plt.grid(True)

    plt.show()

    return pivot

# Analyze Digital Payment Use Cases
def analyze_payment_use_cases(df,date_column="observation_date",value_column="value_numeric"):
    """
    Analyze different digital payment use cases:

    - P2P transfers
    - Merchant payments
    - Bill payments
    - Wage payments
    """

    print("\n========== PAYMENT USE CASE ANALYSIS ==========")

    use_case_indicators = ["USG_P2P_COUNT","MERCHANT_PAYMENT","BILL_PAYMENT","WAGE_PAYMENT"]

    data = df[df["indicator_code"].isin(use_case_indicators)].copy()

    if data.empty:

        print( "No payment use case indicators found.")

        return None

    # Extract year
    data[date_column] = pd.to_datetime(data[date_column])

    data["year"] = (data[date_column].dt.year)

    # Calculate adoption by use case
    use_case_trend = (
        data
        .groupby(
            [
                "year",
                "indicator_code"
            ]
        )[value_column]
        .mean()
        .reset_index()
    )

    print(use_case_trend)

    # Visualization
    plot_data = (
        use_case_trend
        .pivot(
            index="year",
            columns="indicator_code",
            values=value_column
        )
    )

    plot_data.plot(figsize=(10,5),marker="o")

    plt.title("Digital Payment Use Case Adoption")

    plt.xlabel("Year")

    plt.ylabel("Adoption (%)")

    plt.grid(True)

    plt.show()

    return plot_data

def analyze_infrastructure(
        df,
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Analyze infrastructure-related indicators:

    - 4G coverage
    - Mobile penetration
    - ATM density

    """

    print("\n========== INFRASTRUCTURE ANALYSIS ==========")

    infrastructure_indicators = ["ACC_4G_COV","ACC_MOBILE_PEN","USG_ATM_VALUE"]

    infra = df[df["indicator_code"].isin(infrastructure_indicators)].copy()

    if infra.empty:

        print( "No infrastructure data found.")

        return None

    # Convert date
    infra[date_column] = pd.to_datetime(infra[date_column])

    infra["year"] = (infra[date_column].dt.year)

    # Yearly infrastructure trend
    trend = (
        infra
        .groupby(
            [
                "year",
                "indicator_code"
            ]
        )[value_column]
        .mean()
        .reset_index()
    )

    print(trend)

    # Visualization
    plot_data = trend.pivot(index="year",columns="indicator_code",values=value_column)

    plot_data.plot(figsize=(10,5),marker="o")

    plt.title("Financial Inclusion Infrastructure Trends")

    plt.xlabel("Year")

    plt.ylabel("Value")

    plt.grid(True)

    plt.show()

    return plot_data

#  Analyze Infrastructure vs Inclusion Outcomes
def analyze_infrastructure_relationships( df,value_column="value_numeric"):
    """
    Examine relationships between infrastructure
    indicators and financial inclusion outcomes.

    Uses correlation analysis.
    """

    print("\n========== INFRASTRUCTURE RELATIONSHIPS ==========")

    selected_indicators = ["ACC_4G_COV","ACC_MOBILE_PEN","USG_ATM_VALUE",
                           "ACC_OWNERSHIP","DIGITAL_PAYMENT","ACC_MM_ACCOUNT"]

    data = df[
        df["indicator_code"]
        .isin(selected_indicators)
    ].copy()

    if data.empty:

        print("Required indicators not available." )

        return None

    # Convert indicator values into columns
    matrix = (
        data
        .pivot_table(
            index="observation_date",
            columns="indicator_code",
            values=value_column,
            aggfunc="mean"
        )
    )

    correlation = (matrix.corr().round(2))

    print("\nCorrelation Matrix:")

    print(correlation)

    # Plot relationship heatmap
    plt.figure(figsize=(10,7))

    plt.imshow(correlation,cmap="coolwarm")

    plt.colorbar()

    plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90)

    plt.yticks(range(len(correlation.columns)),correlation.columns)

    plt.title("Infrastructure and Inclusion Correlations")

    plt.tight_layout()

    plt.show()

    return correlation

#  Identify Leading Indicators
def identify_leading_indicators(correlation_matrix,target_indicator="ACC_MM_ACCOUNT"):
    """
    Identify infrastructure variables that may
    predict future Findex outcomes.

    """

    print("\n========== LEADING INDICATORS ==========")

    if target_indicator not in correlation_matrix.columns:

        print("Target indicator not found.")

        return None

    predictors = (correlation_matrix[target_indicator].drop(target_indicator)
        .sort_values(ascending=False))

    print("Potential predictors:")

    print(predictors)

    return predictors

def plot_event_timeline(events_df,date_column="observation_date",event_column="category"):
    """
    Plot all cataloged events on a timeline.
    """

    print("\n========== EVENT TIMELINE ==========")

    events = events_df.copy()

    events[date_column] = pd.to_datetime(events[date_column])

    events = (events.sort_values(date_column))

    plt.figure(figsize=(12,6))

    y_position = range(len(events))

    plt.scatter(events[date_column],y_position)

    for i, row in events.iterrows():

        plt.text(row[date_column],list(y_position)[list(events.index).index(i)],
            row[event_column],rotation=45,ha="right")

    plt.yticks([])

    plt.title("Ethiopia Financial Inclusion Events Timeline")

    plt.xlabel("observation_date")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    return events

#  Overlay Events on Indicator Trends
def plot_indicator_with_events(
        indicator_df,
        events_df,
        indicator_code,
        date_column="observation_date",
        value_column="value_numeric",
        event_date_column="observation_date",
        event_column="indicator"):
    """
    Plot indicator trend and overlay events.
    """
    print("\n========== INDICATOR WITH EVENTS ==========")

    # Select indicator
    indicator = indicator_df[indicator_df["indicator_code"]==indicator_code].copy()

    if indicator.empty:

        print("Indicator not found.")

        return None

    # Convert dates
    indicator[date_column] = pd.to_datetime(indicator[date_column])

    events = events_df.copy()

    events[event_date_column] = pd.to_datetime(events[event_date_column])

    # Sort indicator data
    indicator = indicator.sort_values(date_column)

    # Plot indicator
    plt.figure(figsize=(12,6))

    plt.plot(indicator[date_column],indicator[value_column],marker="o",label=indicator_code)

    # Add events
    for _, event in events.iterrows():

        plt.axvline(event[event_date_column],linestyle="--")

        plt.text(event[event_date_column],indicator[value_column].max(),
            event[event_column],rotation=90,verticalalignment="top")

    plt.title(f"{indicator_code} Trend with Events")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# Analyze Specific Event Impact Windows
def analyze_event_windows(
        indicator_df,
        events_df,
        event_names,
        indicator_code,
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Compare indicator changes before and after
    important events.
    """

    print("\n========== EVENT IMPACT ANALYSIS ==========")

    indicator = indicator_df[indicator_df["indicator_code"]==indicator_code].copy()

    indicator[date_column] = pd.to_datetime(indicator[date_column])

    for event_name in event_names:

        event = events_df[events_df["indicator"]==event_name]

        if event.empty:

            print(f"{event_name}: Not found")

            continue

        event_date = pd.to_datetime(event.iloc[0]["date"])

        before = indicator[indicator[date_column]<event_date][value_column].mean()

        after = indicator[indicator[date_column]>= event_date][value_column].mean()

        change = after - before

        print(f"\nEvent: {event_name}")

        print("Before average:",round(before,2))

        print("After average:",round(after,2))

        print("Difference:",round(change,2))

    return True

def analyze_indicator_correlations(
        df,
        value_column="value_numeric",
        date_column="observation_date"):
    """
    Examine correlations between financial inclusion indicators.
    """

    print("\n========== INDICATOR CORRELATIONS ==========")

    # Convert dates
    df = df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    # Create indicator matrix
    indicator_matrix = (
        df
        .pivot_table(
            index=date_column,
            columns="indicator_code",
            values=value_column,
            aggfunc="mean"
        )
    )


    # Calculate correlations
    correlation_matrix = (indicator_matrix.corr().round(2))

    print(correlation_matrix)

    # Visualization
    plt.figure(figsize=(12,8))

    plt.imshow(correlation_matrix,cmap="coolwarm")

    plt.colorbar()

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )


    plt.yticks(range(len(correlation_matrix.columns)),correlation_matrix.columns )

    plt.title("Financial Inclusion Indicator Correlations")

    plt.tight_layout()

    plt.show()

    return correlation_matrix


#  Identify Factors Associated with Access and Usage
def identify_access_usage_factors(
        correlation_matrix,
        access_indicator="ACCOUNT_OWNERSHIP",
        usage_indicator="DIGITAL_PAYMENT"):
    """
    Identify indicators strongly associated with:
    - Access
    - Usage
    """

    print("\n========== ACCESS FACTORS ==========")

    if access_indicator in correlation_matrix.columns:

        access_factors = (correlation_matrix[access_indicator].drop(access_indicator)
            .sort_values(ascending=False))

        print(access_factors)

    else:

        access_factors = None

        print("Access indicator not found.")

    print("\n========== USAGE FACTORS ==========")

    if usage_indicator in correlation_matrix.columns:

        usage_factors = (correlation_matrix[usage_indicator].drop(usage_indicator)
            .sort_values( ascending=False))

        print(usage_factors)

    else:

        usage_factors = None

        print("Usage indicator not found.")

    return {"access_factors": access_factors,"usage_factors": usage_factors}


#  Analyze Existing Impact Links
def analyze_impact_links(impact_df):
    """
    Examine existing impact_link records.

    Expected columns:
    parent_id
    pillar
    related_indicator
    impact_direction
    impact_magnitude
    lag_months
    evidence_basis
    """

    print("\n========== IMPACT LINK ANALYSIS ==========")

    df = impact_df.copy()

    # Basic information
    print("\nNumber of Impact Links:")

    print(len(df))

    print("\nColumns:")

    print(df.columns.tolist())


    #  Relationship by pillar
    if "pillar" in df.columns:

        print("\nRelationships by Pillar:")

        print(df["pillar"].value_counts(dropna=False))

    #  Impact direction analysis

    if "impact_direction" in df.columns:

        print("\nImpact Direction:")

        print(df["impact_direction"].value_counts(dropna=False))

      # Related indicators
    if "related_indicator" in df.columns:

        print("\nMost Connected Indicators:")

        print(df["related_indicator"].value_counts().head(10))

       # Impact magnitude handling
    if "impact_magnitude" in df.columns:

        print("\nImpact Magnitude Values:")

        print(df["impact_magnitude"].unique())

        # Try converting to numeric
        numeric_magnitude = pd.to_numeric(df["impact_magnitude"],errors="coerce")

        # If numeric values exist
        if numeric_magnitude.notna().any():

            df["impact_magnitude_numeric"] = (numeric_magnitude)

            print("\nAverage Impact Magnitude:")

            print(df.groupby("related_indicator")["impact_magnitude_numeric"]
                .mean().sort_values(ascending=False))

        else:

            print("\nImpact magnitude is categorical.")

            print(df.groupby("related_indicator")["impact_magnitude"].value_counts())

    return df