import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#  Load Impact Links and Event Data
def load_event_impact_data(
        file_path,
        data_sheet="data",
        impact_sheet="impact_links"):
    """
    Load events and impact_links sheets.
    """
    print("\n========== LOADING DATA ==========")

    # Load sheets
    data_df = pd.read_excel(file_path,sheet_name=data_sheet)

    impact_df = pd.read_excel(file_path,sheet_name=impact_sheet)

    print("Data records:", len(data_df))

    print("Impact links:", len(impact_df))

    return data_df, impact_df

#  Join Events with Impact Links
def join_events_with_impacts(
        data_df,
        impact_df,
        event_id_column="record_id",
        parent_id_column="parent_id"):
    """
    Join events with impact_links using parent_id.

    Parameters
    ----------
    data_df:
        Main data sheet containing events

    impact_df:
        impact_links sheet

    event_id_column:
        Event identifier column in data sheet

    parent_id_column:
        Linking column in impact_links
    """

    print("\n========== JOIN EVENT IMPACT DATA ==========")

    # Select events only
    events = data_df[data_df["record_type"] == "impact_link"].copy()


    print("Events found:",len(events))

    # Check columns exist
    if event_id_column not in events.columns:

        raise ValueError(
            f"{event_id_column} not found in data sheet. "
            f"Available columns: {events.columns.tolist()}"
        )

    if parent_id_column not in impact_df.columns:

        raise ValueError(
            f"{parent_id_column} not found in impact_links. "
            f"Available columns: {impact_df.columns.tolist()}"
        )

    # Merge
    event_impacts = impact_df.merge(
        events,
        left_on=parent_id_column,
        right_on=event_id_column,
        how="left",
        suffixes=("_impact", "_event")
    )

    print("Joined records:",len(event_impacts))

    return event_impacts

# Create Event Impact Summary
def create_event_impact_summary(event_impacts):
    """
    Create summary showing:
    Event
       |
       ↓
    Indicator affected
       |
       ↓
    Impact direction and magnitude
    """

    print("\n========== EVENT IMPACT SUMMARY ==========")

    summary_columns = ["category_event","related_indicator","pillar","impact_direction",
                        "impact_magnitude","lag_months","evidence_basis"]

    # Keep available columns only
    available_columns = [
        col for col in summary_columns
        if col in event_impacts.columns
    ]

    summary = (event_impacts[available_columns].sort_values( by="category_event" ) )

    print(summary)

    return summary

#  Create Event Impact Summary
def create_event_effect_curve(event_date,end_date,lag_months,impact_strength):

    # Convert strings to datetime
    event_date = pd.to_datetime(event_date)
    end_date = pd.to_datetime(end_date)

    dates = pd.date_range(start=event_date,end=end_date,freq="MS")

    effects = []

    for date in dates:

        months_after = ((date.year - event_date.year) * 12 + (date.month - event_date.month))

        if months_after < lag_months:
            effect = 0
        else:
            effect = impact_strength * (1 - np.exp(-(months_after - lag_months) / 12))

        effects.append(effect)

    return pd.DataFrame({"date": dates,"event_effect": effects})

#  Prepare Event Impact Links
def prepare_event_impacts(impact_df):
    """
    Prepare impact links for modeling.

    Required columns:
    parent_id
    related_indicator
    impact_direction
    impact_magnitude
    lag_months
    """

    print("\n========== PREPARING EVENT IMPACTS ==========")

    df = impact_df.copy()

    # Convert impact direction to numerical effect
    direction_map = {"positive": 1,"negative": -1,"neutral": 0}

    df["direction_score"] = (df["impact_direction"].map(direction_map).fillna(0))

    # Convert impact magnitude
    magnitude_map = {"high": 3,"medium": 2,"low": 1}

    df["magnitude_score"] = (df["impact_magnitude"].map(magnitude_map).fillna(1))

    # Calculate total impact strength
    df["impact_strength"] = (df["direction_score"]*df["magnitude_score"])

    print(df.head())

    return df


#  Create Time-Based Event Effects
def create_event_effect_curve(event_date,end_date,lag_months,impact_strength):
    """
    Represent how event impact changes over time.

    Assumption:
    - No effect during lag period
    - Gradual increase after lag
    """

    # Convert to datetime if necessary
    event_date = pd.to_datetime(event_date)
    end_date = pd.to_datetime(end_date)

    dates = pd.date_range(
        start=event_date,
        end=end_date,
        freq="MS"
    )

    effects = []

    for current_date in dates:

        months_after = (
            (current_date.year - event_date.year) * 12 +
            (current_date.month - event_date.month)
        )

        if months_after < lag_months:
            effect = 0

        else:
            effect = impact_strength * (
                1 - np.exp(-(months_after - lag_months) / 12)
            )

        effects.append(effect)

    return pd.DataFrame({
        "date": dates,
        "effect": effects
    })

#  Combine Multiple Event Effects
def combine_event_effects(event_curves):
    """
    Combine effects from multiple events.
    """

    print("\n========== COMBINING EVENT EFFECTS ==========")

    combined = (event_curves.groupby("date")["effect"].sum().reset_index())

    return combined


#  Predict Indicator Change
def predict_indicator_change(historical_df,event_effect_df,
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Add event effects to historical indicator values.
    """

    print("\n========== PREDICTED INDICATOR CHANGE ==========")

    df = historical_df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    merged = df.merge(event_effect_df,left_on=date_column,
        right_on="date",how="left")

    merged["effect"] = (merged["effect"].fillna(0))

    # predicted value
    merged["predicted_value"] = (merged[value_column] + merged["effect"])

    plt.figure(figsize=(10,5))

    plt.plot(merged[date_column],merged[value_column],label="Observed")

    plt.plot(merged[date_column], merged["predicted_value"],label="Event Adjusted")

    plt.title("Indicator Change After Events")

    plt.xlabel("Date")

    plt.ylabel("Indicator Value")

    plt.legend()

    plt.grid(True)

    plt.show()

    return merged

# =====================================================
# Create the Association Matrix: 
# =====================================================

# Convert Impact to Numerical Effect
def prepare_event_impact_matrix(data_df,impact_df,event_id_col="record_id",
                                parent_id_col="parent_id"):
    """
    Join events with impact links.
    Output:
    Event + indicator + impact information
    """

    print("\n========== PREPARING EVENT IMPACT DATA ==========")

    # Select event records
    events = data_df[data_df["record_type"].str.lower()=="event"].copy()

    print("Number of events:",len(events))

    # Join events with impact links
    event_impact = impact_df.merge(events,left_on=parent_id_col,right_on=event_id_col,
                                   how="left")

    print("Joined impact records:",len(event_impact) )

    return event_impact


# Function 2: Convert Impact to Numerical Effect
def calculate_event_effect(event_impact):
    """
    Convert qualitative impact information
    into numerical scores.

    Positive high impact = +3
    Positive medium = +2
    Positive low = +1
    Negative effects are negative values.
    """

    print("\n========== CALCULATING EFFECT SCORES ==========")

    df = event_impact.copy()

    # Direction mapping
    direction_map = {"positive": 1, "negative": -1, "neutral": 0}

    # Magnitude mapping
    magnitude_map = { "high": 3, "medium": 2, "low": 1}

    # Check available columns
    if "impact_direction" in df.columns:
        df["direction_score"] = (
            df["impact_direction"]
            .str.lower()
            .map(direction_map)
            .fillna(0)
        )
    else:
        df["direction_score"] = 1

    if "impact_magnitude" in df.columns:

        df["magnitude_score"] = (df["impact_magnitude"].str.lower().map(magnitude_map)
            .fillna(1))

    else:

        df["magnitude_score"] = 1

    # Estimated effect
    df["estimated_effect"] = (df["direction_score"] * df["magnitude_score"])

    return df

# Function 3: Build Association Matrix
def create_association_matrix(event_impact_df,event_column="category_x",
        indicator_column="related_indicator_x",effect_column="estimated_effect"):
    """
    Create Event-Indicator Association Matrix.
    Rows:
        Events
    Columns:
        Indicators
    Values:
        Estimated impact
    """

    print("\n========== ASSOCIATION MATRIX ==========")

    matrix = pd.pivot_table(event_impact_df,index=event_column,columns=indicator_column,
                            values=effect_column,aggfunc="sum",fill_value=0)

    print(matrix)

    return matrix

# =====================================================
# Test Your Model Against Historical Data
# =====================================================


# Function 1: Prepare Historical Indicator Data
def prepare_indicator_history(df,indicator_code, date_column="observation_date",
                              value_column="value_numeric"):
    """
    Extract historical indicator trend.
    """

    print("\n========== PREPARING HISTORICAL DATA ==========")

    indicator_df = df[df["indicator_code"] == indicator_code].copy()

    indicator_df[date_column] = pd.to_datetime(indicator_df[date_column])

    indicator_df["year"] = (indicator_df[date_column].dt.year)

    yearly_data = (indicator_df.groupby("year")[value_column].mean().reset_index())

    print(yearly_data)

    return yearly_data


# Function 2: Calculate Actual Event Impact
def calculate_actual_event_change(yearly_data,event_year=2021,value_column="value_numeric"):
    """
    Calculate real indicator change after event.
    """

    print("\n========== ACTUAL EVENT CHANGE ==========")

    before = yearly_data[yearly_data["year"] == event_year][value_column].values[0]

    after = yearly_data[yearly_data["year"] == 2024][value_column].values[0]

    actual_change = after - before

    print("Value before event:",before)

    print("Value after event:",after)

    print("Actual change:",round(actual_change,2),"percentage points")

    return actual_change


# Function 3: Compare Model Impact vs Reality
def compare_model_with_actual(estimated_effect,actual_change):
    """
    Compare predicted impact with observed change.
    """

    print("\n========== MODEL VALIDATION ==========")

    difference = (actual_change-estimated_effect)

    print("Model estimated impact:",estimated_effect)

    print("Observed impact:",actual_change)

    print("Difference:",round(difference,2))


    if abs(difference) <= 1:

        conclusion = ("Model impact is consistent with historical data.")

    else:

        conclusion = ("Model impact differs from observed change.")

    print( conclusion)

    return difference


# Function 4: Explain Model Difference
def explain_difference():
    """
    Possible reasons why model and reality differ.
    """

    print("\n========== POSSIBLE EXPLANATIONS ==========")


    explanations = [

        "Other events occurred at the same time (competition, regulation, infrastructure).",

        "Event effects may have a longer or shorter lag than assumed.",

        "Mobile money registration does not always mean active usage.",

        "Economic conditions may influence adoption.",

        "Regional and demographic differences may hide the real impact.",

        "Impact magnitude assumptions may need calibration."

    ]


    for item in explanations:

        print("-", item)

    return explanations


# =====================================================
# Refine Estimates
# =====================================================

# Function 1: Compare Estimated vs Observed Impact
def evaluate_event_estimates(
        impact_matrix,
        historical_changes):
    """
    Compare model estimated effects with
    observed indicator changes.

    Parameters
    ----------
    impact_matrix:
        Event-indicator estimated effects

    historical_changes:
        Actual observed changes by indicator

    """

    print("\n========== EVALUATING IMPACT ESTIMATES ==========")


    evaluation = []


    for event in impact_matrix.index:

        for indicator in impact_matrix.columns:


            estimated = (
                impact_matrix
                .loc[event, indicator]
            )


            actual = (
                historical_changes
                .get(indicator, None)
            )


            if actual is not None:


                difference = (
                    actual
                    -
                    estimated
                )


                evaluation.append({

                    "event": event,

                    "indicator": indicator,

                    "estimated_effect": estimated,

                    "actual_change": actual,

                    "difference": difference

                })



    evaluation_df = pd.DataFrame(
        evaluation
    )


    print(evaluation_df)


    return evaluation_df




# =====================================================
# Function 2: Refine Impact Estimates
# =====================================================
def refine_impact_estimates(
        evaluation_df):
    """
    Adjust estimates based on observed evidence.

    Rules:
    - Close match -> keep estimate
    - Underestimated -> increase
    - Overestimated -> decrease

    """


    print("\n========== REFINING ESTIMATES ==========")


    refined = evaluation_df.copy()



    def adjust(row):

        difference = row["difference"]


        if abs(difference) <= 1:

            return row["estimated_effect"]



        elif difference > 1:

            # model underestimated

            return row["estimated_effect"] + 1



        else:

            # model overestimated

            return row["estimated_effect"] - 1



    refined["refined_effect"] = (
        refined
        .apply(
            adjust,
            axis=1
        )
    )


    print(refined)


    return refined




# =====================================================
# Function 3: Document Reasoning and Confidence
# =====================================================
def create_estimate_documentation(
        refined_df):
    """
    Create explanation for adjustments.

    """

    print("\n========== IMPACT DOCUMENTATION ==========")



    documentation = []



    for _, row in refined_df.iterrows():


        if (
            abs(row["difference"])
            <= 1
        ):

            confidence = "High"

            reason = (
                "Estimated impact aligns with historical observation."
            )


        elif (
            abs(row["difference"])
            <= 3
        ):

            confidence = "Medium"

            reason = (
                "Some difference exists; external factors may influence results."
            )


        else:

            confidence = "Low"

            reason = (
                "Large difference; estimate requires further investigation."
            )



        documentation.append({

            "event": row["event"],

            "indicator": row["indicator"],

            "old_estimate": row["estimated_effect"],

            "new_estimate": row["refined_effect"],

            "confidence": confidence,

            "reason": reason

        })



    documentation_df = pd.DataFrame(
        documentation
    )


    print(documentation_df)


    return documentation_df