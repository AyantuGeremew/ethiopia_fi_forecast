import pandas as pd

# =====================================================
# 1. Add Additional Observations
# =====================================================
def add_observations(df, new_observations):
    """
    Add additional financial inclusion observations.

    Examples:
    - Gender disaggregated access
    - Regional financial access
    - Infrastructure coverage
    - Findex indicators

    Required columns example:
    date
    record_type
    indicator_code
    value
    source
    """

    print("\n========== ADDING OBSERVATIONS ==========")

    new_observations["record_type"] = "observation"

    updated_df = pd.concat(
        [
            df,
            new_observations
        ],
        ignore_index=True
    )

    print(
        "New observations added:",
        len(new_observations)
    )

    print(
        "Total records:",
        len(updated_df)
    )

    return updated_df



# =====================================================
# 2. Add Additional Events
# =====================================================
def add_events(df, new_events):
    """
    Add new events affecting financial inclusion.

    Examples:
    - New financial policies
    - Digital banking launches
    - Mobile money expansion
    - Infrastructure investments
    - Regulatory changes

    Required columns example:
    date
    event
    pillar
    source
    """

    print("\n========== ADDING EVENTS ==========")

    new_events["record_type"] = "event"

    updated_df = pd.concat(
        [
            df,
            new_events
        ],
        ignore_index=True
    )

    print(
        "New events added:",
        len(new_events)
    )

    print(
        "Total records:",
        len(updated_df)
    )

    return updated_df



# =====================================================
# 3. Add Impact Links
# =====================================================
def add_impact_links(
        impact_df,
        new_links):
    """
    Add new relationships between events
    and financial inclusion indicators.

    Example:

    Event:
        Mobile Money Launch

    Indicator:
        DIGITAL_PAYMENT

    Impact:
        Positive

    Strength:
        0.80

    """

    print("\n========== ADDING IMPACT LINKS ==========")

    updated_links = pd.concat(
        [
            impact_df,
            new_links
        ],
        ignore_index=True
    )

    print(
        "New impact links added:",
        len(new_links)
    )

    print(
        "Total impact links:",
        len(updated_links)
    )

    return updated_links