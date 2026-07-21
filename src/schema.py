import pandas as pd
from datetime import datetime


# ==========================================================
# 1. Add New Observation
# ==========================================================
def add_observation(
    observations_df,
    pillar,
    indicator,
    indicator_code,
    value_numeric,
    observation_date,
    source_name,
    source_url,
    original_text,
    confidence,
    collected_by,
    notes
):
    """
    Add a new observation record.
    """

    new_record = {
        "record_type": "observation",
        "pillar": pillar,
        "indicator": indicator,
        "indicator_code": indicator_code,
        "value_numeric": value_numeric,
        "observation_date": observation_date,
        "source_name": source_name,
        "source_url": source_url,
        "original_text": original_text,
        "confidence": confidence,
        "collected_by": collected_by,
        "collection_date": datetime.today().strftime("%Y-%m-%d"),
        "notes": notes
    }

    observations_df = pd.concat(
        [observations_df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    print("✓ Observation added successfully.")

    return observations_df


# ==========================================================
# 2. Add New Event
# ==========================================================
def add_event(
    events_df,
    event_name,
    event_date,
    category,
    source_name,
    source_url,
    original_text,
    confidence,
    collected_by,
    notes
):
    """
    Add a new event record.

    Pillar is intentionally left blank.
    """

    new_record = {
        "record_type": "event",
        "pillar": "",
        "event": event_name,
        "event_date": event_date,
        "category": category,
        "source_name": source_name,
        "source_url": source_url,
        "original_text": original_text,
        "confidence": confidence,
        "collected_by": collected_by,
        "collection_date": datetime.today().strftime("%Y-%m-%d"),
        "notes": notes
    }

    events_df = pd.concat(
        [events_df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    print("✓ Event added successfully.")

    return events_df


# ==========================================================
# 3. Add New Impact Link
# ==========================================================
def add_impact_link(
    impact_links_df,
    parent_id,
    pillar,
    related_indicator,
    impact_direction,
    impact_magnitude,
    lag_months,
    evidence_basis,
    source_name,
    source_url,
    original_text,
    confidence,
    collected_by,
    notes
):
    """
    Add a relationship between an event and an indicator.
    """

    new_record = {
        "parent_id": parent_id,
        "pillar": pillar,
        "related_indicator": related_indicator,
        "impact_direction": impact_direction,
        "impact_magnitude": impact_magnitude,
        "lag_months": lag_months,
        "evidence_basis": evidence_basis,
        "source_name": source_name,
        "source_url": source_url,
        "original_text": original_text,
        "confidence": confidence,
        "collected_by": collected_by,
        "collection_date": datetime.today().strftime("%Y-%m-%d"),
        "notes": notes
    }

    impact_links_df = pd.concat(
        [impact_links_df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    print("✓ Impact link added successfully.")

    return impact_links_df