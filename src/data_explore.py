import pandas as pd


# =====================================================
# 1. Count Records 
# =====================================================

# Count Records by Record Type
def count_by_record_type(df, column="record_type"):
    """
    Count records by type.

    Examples:
    - observation
    - event
    - target
    """

    print("\n========== RECORD TYPE COUNT ==========")

    if column not in df.columns:
        print(f"{column} column not found")
        return None


    result = (df[column].value_counts(dropna=False).reset_index())

    result.columns = ["Record Type","Count"]

    print(result)

    return result

# Count Records by Pillar
def count_by_pillar(df, column="pillar"):
    """
    Count records by financial inclusion pillar.

    Examples:
    - access
    - usage
    - quality
    - digital_finance
    """

    print("\n========== PILLAR COUNT ==========")

    if column not in df.columns:
        print(f"{column} column not found")
        return None

    result = (df[column].value_counts(dropna=False).reset_index())

    result.columns = ["Pillar","Count"]

    print(result)

    return result

# Count Records by Source Type
def count_by_source_type(df, column="source_type"):
    """
    Count records by data source.

    Examples:
    - survey
    - report
    - operator
    - government
    """

    print("\n========== SOURCE TYPE COUNT ==========")

    if column not in df.columns:
        print(f"{column} column not found")
        return None

    result = (df[column].value_counts(dropna=False).reset_index())

    result.columns = ["Source Type","Count"]

    print(result)

    return result


# Count Records by Confidence Level
def count_by_confidence(df, column="confidence"):
    """
    Count records by confidence level.

    Examples:
    - high
    - medium
    - low
    """

    print("\n========== CONFIDENCE COUNT ==========")

    if column not in df.columns:
        print(f"{column} column not found")
        return None

    result = (df[column].value_counts(dropna=False).reset_index())

    result.columns = ["Confidence","Count"]

    print(result)

    return result


# =====================================================
# Identify Temporal Range of Observations
# =====================================================
def identify_observation_temporal_range(df,record_column="record_type",date_column="collected_by"):
    """
    Identify the temporal range of observation records.
    Parameters
    ----------
    df : pandas.DataFrame
        Financial inclusion dataset
    record_column : str
        Column containing record categories
        (observation, event, target)
    date_column : str
        Date column name
    Returns
    -------
    dict
        Observation time range summary
    """

    print("\n========== OBSERVATION TEMPORAL ANALYSIS ==========")

    # Filter observation records
    observations = df[
        df[record_column]
        .astype(str)
        .str.lower()
        .str.strip()
        == "observation"
    ].copy()

    if observations.empty:
        print("No observation records found.")
        return None

    # Convert date column
    observations[date_column] = pd.to_datetime(observations[date_column])

    # Calculate temporal range
    start_date = observations[date_column].min()

    end_date = observations[date_column].max()

    duration_days = (end_date - start_date).days

    duration_years = round(duration_days / 365,2)

    # Observation frequency
    yearly_count = (observations[date_column].dt.year.value_counts().sort_index())

    #  Display Results
    print(f"Total Observations : {len(observations)}")

    print(f"Start Date         : {start_date.date()}")

    print(f"End Date           : {end_date.date()}")

    print(f"Duration (Days)    : {duration_days}")

    print(f"Duration (Years)   : {duration_years}")

    print("\nObservation Count by Year:")

    print(yearly_count)

    # Return Results
    return {
        "total_observations": len(observations),
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
        "duration_years": duration_years,
        "yearly_frequency": yearly_count
    }

# =====================================================
# List Unique Indicators and Their Coverage
# =====================================================
def analyze_indicator_coverage(df,indicator_column="indicator_code"):
    """
    Analyze unique indicators and their data coverage.

    Parameters
    ----------
    df : pandas.DataFrame
        Financial inclusion dataset

    indicator_column : str
        Column containing indicator codes

    Returns
    -------
    pandas.DataFrame
        Indicator coverage summary
    """

    print("\n========== INDICATOR COVERAGE ANALYSIS ==========")

    # Check indicator column
    if indicator_column not in df.columns:
        raise ValueError(
            f"{indicator_column} column not found"
        )

    # Count records per indicator
    indicator_summary = (df[indicator_column].value_counts(dropna=False).reset_index())

    indicator_summary.columns = ["Indicator Code","Record Count"]

    # Calculate coverage percentage
    total_records = len(df)

    indicator_summary["Coverage (%)"] = (indicator_summary["Record Count"] / total_records
        * 100
    ).round(2)

    # Display results
    print("Total Unique Indicators:",df[indicator_column].nunique())

    print("\nIndicator Coverage:")

    print(indicator_summary)

    return indicator_summary


# =====================================================
# Review Cataloged Events and Dates
# =====================================================
def review_events(df,event_column="category",date_column="collection_date",record_column="record_type"):
    """
    Review events recorded in the dataset.

    Shows:
    - Total number of events
    - Event names
    - Event dates
    - Event frequency
    """

    print("\n========== CATALOGED EVENTS ==========")

    # Filter event records
    events = df[
        df[record_column]
        .astype(str)
        .str.lower()
        .str.strip()
        == "category"
    ].copy()

    if events.empty:
        print("No event records found.")
        return None

    # Convert date
    events[date_column] = pd.to_datetime(events[date_column])

    print("Total Events:",len(events))

    print("\nEvent List and Dates:")

    event_summary = events[
        [
            event_column,
            date_column
        ]
    ].sort_values(
        by=date_column
    )

    print(event_summary)

    return events


# =====================================================
#  Review Impact Links Relationships
# =====================================================
def review_impact_links(impact_df,event_column="parent_id",indicator_column="indicator_code"):
    """
    Review relationships between events and indicators.

    Shows:
    - Number of impact relationships
    - Events connected to indicators
    - Indicators affected by events
    """

    print("\n========== IMPACT LINKS REVIEW ==========")

    print("Total Impact Links:",len(impact_df))

    # Display relationship columns
    print("\nAvailable Columns:")

    for col in impact_df.columns:
        print("-", col)

    # Check relationship fields
    if (event_column in impact_df.columns and indicator_column in impact_df.columns):

        relationships = (
            impact_df[
                [
                    event_column,
                    indicator_column
                ]
            ]
            .drop_duplicates()
        )

        print("\nEvent-Indicator Relationships:")

        print(relationships)

        print("\nNumber of Unique Relationships:",len(relationships))

        return relationships

    else:

        print("Required relationship columns not found.")

        return None