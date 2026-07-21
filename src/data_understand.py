import pandas as pd



# =====================================================
# 1. Examine the dataset structure
# =====================================================

# Check Dataset Dimensions
def check_dimensions(df):
    """
    Check number of rows and columns.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    None
    """

    print("\n========== DATASET DIMENSION ==========")

    rows, columns = df.shape

    print(f"Total Records : {rows}")
    print(f"Total Columns : {columns}")


# Display Column Names
def display_columns(df):
    """
    Display all columns in dataset.
    """

    print("\n========== DATASET COLUMNS ==========")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")


#  Check Column Consistency
def check_column_consistency(df):
    """
    Verify that all records share the same columns.

    Returns
    -------
    Boolean
    """

    print("\n========== COLUMN CONSISTENCY CHECK ==========")
    # Expected columns from first record
    first_record_columns = set(df.iloc[0].index)
    inconsistent_records = []
    for index, row in df.iterrows():
        current_columns = set(row.index)
        if current_columns != first_record_columns:
            inconsistent_records.append(index)
    if len(inconsistent_records) == 0:
        print("✓ All records have the same columns.")
        return True
    else:
        print("✗ Inconsistent records found.")
        print(
            f"Number of inconsistent records: {len(inconsistent_records)}"
        )
        print(
            "Records:",
            inconsistent_records[:10]
        )
        return False


#  Show Column Data Types
def show_column_types(df):
    """
    Display column names and data types.
    """

    print("\n========== COLUMN DATA TYPES ==========")

    structure = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.values

    })
    print(structure)


#  Generate Dataset Schema
def generate_schema(df):
    """
    Generate dataset schema information.
    """
    print("\n========== DATASET SCHEMA ==========")
    schema = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values
    })
    return schema


# Complete Structure Analysis
def examine_structure(df):
    """
    Run complete dataset structure examination.
    """
    check_dimensions(df)
    display_columns(df)
    show_column_types(df)
    check_column_consistency(df)
    schema = generate_schema(df)
    print("\nSchema Report:")
    print(schema)
    return schema

# =====================================================
# 2 . understanding the sheet 1 data
# =====================================================

# Check Available Record Types
def check_record_types(df, type_column="record_type"):
    """
    Display unique record categories.
    """

    print("\n========== AVAILABLE RECORD TYPES ==========")

    if type_column not in df.columns:
        print(f"Column '{type_column}' does not exist.")
        return

    types = df[type_column].unique()

    for t in types:
        print("-", t)


#  Count Records by Type
def count_records_by_type(df, type_column="record_type"):
    """
    Count number of records in each category.
    """

    print("\n========== RECORD COUNTS ==========")

    counts = df[type_column].value_counts()

    print(counts)

    return counts


# Extract Observation Records
def analyze_observations(df, type_column="record_type"):
    """
    Analyze actual measured financial inclusion observations.

    Examples:
    - Survey measurements
    - Reports
    - Operator data
    """

    print("\n========== OBSERVATION RECORDS ==========")

    observations = df[df[type_column].str.lower() == "observation"]

    print("Number of observations:",
          len(observations))

    print("\nSample observations:")

    print(observations.head())

    return observations

#  Extract Event Records
def analyze_events(df, type_column="record_type"):
    """
    Analyze events affecting financial inclusion.

    Examples:
    - Policies
    - Product launches
    - Market entries
    - Milestones
    """

    print("\n========== EVENT RECORDS ==========")

    events = df[df[type_column].str.lower() == "event"]

    print("Number of events:",
          len(events))

    print("\nSample events:")

    print(events.head())

    return events

# Extract Target Records
def analyze_targets(df, type_column="record_type"):
    """
    Analyze official policy targets.

    Examples:
    - Government financial inclusion goals
    - Strategic targets
    """

    print("\n========== TARGET RECORDS ==========")


    targets = df[df[type_column].str.lower() == "target"]

    print("Number of targets:",len(targets))

    print("\nSample targets:")

    print(targets.head())

    return targets

#  Compare Record Categories
def compare_record_categories(df, type_column="record_type"):
    """
    Compare observations, events, and targets.
    """

    print("\n========== CATEGORY COMPARISON ==========")

    summary = (df.groupby(type_column).size().reset_index(name="Number of Records"))

    print(summary)

    return summary


# =====================================================
# 3. Understand sheet 2 data
# =====================================================

# Examine Impact Link Structure
def examine_impact_structure(df):
    """
    Understand columns, records, and data types.
    """

    print("\n========== IMPACT LINKS STRUCTURE ==========")

    print("Number of Records:", len(df))

    print("Number of Columns:", len(df.columns))

    print("\nColumns:")

    for col in df.columns:
        print("-", col)

    print("\nData Types:")

    print(df.dtypes)

# Summarize Event-Indicator Relationships
def summarize_relationships(df):
    """
    Display relationship summary.

    Expected columns may include:
    event
    indicator
    impact
    strength
    confidence
    """

    print("\n========== RELATIONSHIP SUMMARY ==========")

    print(df.head())

    print("\nMissing Values:")

    print(df.isnull().sum())

# Analyze Impact Direction and Strength
def analyze_impact(df):

    """
    Analyze modeled impacts.

    Positive impact:
        Event improves indicator

    Negative impact:
        Event reduces indicator
    """

    print("\n========== IMPACT ANALYSIS ==========")

    # Display columns for checking
    print("Available columns:")

    print(list(df.columns))

    # Example if impact column exists

    if "impact" in df.columns:

        print("\nImpact Distribution:")

        print(df["impact"].value_counts())

    # Example if strength column exists

    if "strength" in df.columns:

        print("\nAverage Impact Strength:")

        print(df["strength"].mean())

# =====================================================
# 4. Review reference code data Valid Field Values
# =====================================================
def review_valid_values(df):
    """
    Display unique valid values for each field.
    
    Used for data validation.
    """

    print("\n========== VALID FIELD VALUES ==========")

    for column in df.columns:

        print("\nField:", column)

        values = df[column].dropna().unique()

        print("Valid Values:")

        for value in values:
            print("-", value)

        print("Total Valid Values:",len(values))

# =====================================================
# 
# =====================================================


# Analyze Event Structure and Pillar Values
def analyze_event_pillars(df, event_type="event"):
    """
    Understand how events are categorized using pillars.
    Examples:
    - policy
    - product_launch
    - market_entry
    - milestone
    """

    print("\n========== EVENT PILLAR ANALYSIS ==========")

    # Filter event records

    events = df[ df["record_type"].str.lower() == event_type]

    print("Total Events:", len(events))

    print("\nAvailable Columns:")

    print(list(events.columns))

    # Check pillar column

    if "pillar" in events.columns:

        print("\nPillar Distribution:")

        print(events["pillar"].value_counts(dropna=False))

    else:

        print("\nNo pillar column found.")

    return events

#  Identify Challenges in Pillar Assignment
def check_pillar_challenges(events):
    """
    Identify issues when assigning pillars to events.
    Challenges:
    - Missing pillars
    - Multiple possible categories
    - Inconsistent naming
    """

    print("\n========== PILLAR ASSIGNMENT CHALLENGES ==========")

    if "pillar" not in events.columns:

        print("Pillar field does not exist.")

        return

    # Missing values

    missing = events["pillar"].isnull().sum()

    print("Missing pillar values:",missing)

    # Unique naming

    print("\nExisting pillar values:" )

    print(events["pillar"].unique())

    print(
        """
Common Challenges:

1. One event can belong to multiple pillars.
   Example:
   Mobile banking launch:
   - Digital access
   - Product innovation

2. Different sources may use different names.
   Example:
   product launch vs product_launch

3. Some events may have unclear impact category.

4. Historical events may lack pillar information.
        """
    )


# Connect Impact Links using Parent ID
def analyze_parent_id_links(events, impact_links):
    """
    Understand relationship:
    Event(parent_id)
          |
          |
    impact_link
          |
          |
    Indicator
    """

    print("\n========== IMPACT LINK CONNECTION ==========")

    print("Events:",len(events))

    print( "Impact Links:", len(impact_links))

    # Check parent_id

    if "parent_id" not in impact_links.columns:

        print( "parent_id column not found")

        return

    print("\nSample impact connections:")

    connection = impact_links.merge(
        events,
        left_on="record_id",
        right_on="record_id",
        how="left"
    )

    print(connection.head())

    print("\nConnected Records:")

    print(
        connection[
            "parent_id"
        ]
        .notnull()
        .sum()
    )

    return connection