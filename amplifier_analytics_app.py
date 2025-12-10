#!/usr/bin/env python3
"""
Amplifier Analytics App - Cohorts & Funnels

A Streamlit app for analyzing event attendance, cohort retention, churn, and user behavior.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Amplifier Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the three CSV files.
    
    Returns:
        Tuple of (events_df, people_df, attendees_df)
    """
    # Load CSVs with error handling
    try:
        events_df = pd.read_csv("events.csv")
    except FileNotFoundError:
        st.error("❌ Error: events.csv not found. Please ensure the file exists in the same directory.")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading events.csv: {str(e)}")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        people_df = pd.read_csv("people.csv")
    except FileNotFoundError:
        st.error("❌ Error: people.csv not found. Please ensure the file exists in the same directory.")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading people.csv: {str(e)}")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        attendees_df = pd.read_csv("attendees.csv")
    except FileNotFoundError:
        st.error("❌ Error: attendees.csv not found. Please ensure the file exists in the same directory.")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading attendees.csv: {str(e)}")
        st.stop()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Parse datetime columns and normalize to timezone-naive (always CET/Berlin, no timezone conversion needed)
    date_cols_events = ["Start Date & Time", "End Date & Time"]
    date_cols_people = ["First Event Date", "Most Recent Event Date"]
    date_cols_attendees = ["RSVP Time", "Check-in Timestamp", "Event Date"]
    
    # Helper to remove timezone info (always CET/Berlin, no timezone conversion needed)
    def remove_timezone(series):
        """Remove timezone info from datetime series."""
        if series.empty:
            return series
        # Check if any values are timezone-aware
        sample = series.dropna()
        if len(sample) > 0:
            try:
                # Try to access tzinfo - if it exists, the series is timezone-aware
                if sample.iloc[0].tzinfo is not None:
                    # Convert to UTC first, then remove timezone
                    return series.dt.tz_convert('UTC').dt.tz_localize(None)
            except (AttributeError, TypeError):
                pass
        return series
    
    for col in date_cols_events:
        if col in events_df.columns:
            events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
            events_df[col] = remove_timezone(events_df[col])
    
    for col in date_cols_people:
        if col in people_df.columns:
            people_df[col] = pd.to_datetime(people_df[col], errors='coerce')
            people_df[col] = remove_timezone(people_df[col])
    
    for col in date_cols_attendees:
        if col in attendees_df.columns:
            attendees_df[col] = pd.to_datetime(attendees_df[col], errors='coerce')
            attendees_df[col] = remove_timezone(attendees_df[col])
    
    # Normalize emails to lowercase
    if "Email" in people_df.columns:
        people_df["Email"] = people_df["Email"].str.lower()
    if "Email" in attendees_df.columns:
        attendees_df["Email"] = attendees_df["Email"].str.lower()
    
    # Ensure booleans are actually booleans
    bool_cols_attendees = ["Checked In", "Flaked?"]
    for col in bool_cols_attendees:
        if col in attendees_df.columns:
            attendees_df[col] = attendees_df[col].astype(bool)
    
    if "Churned?" in people_df.columns:
        people_df["Churned?"] = people_df["Churned?"].map({"Yes": True, "No": False, "": False})
        people_df["Churned?"] = people_df["Churned?"].fillna(False).astype(bool)
    
    return events_df, people_df, attendees_df


def apply_filters(
    events_df: pd.DataFrame,
    people_df: pd.DataFrame,
    attendees_df: pd.DataFrame,
    date_range: Tuple[datetime, datetime],
    event_types: List[str],
    topic_categories: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply sidebar filters to the dataframes.
    
    Args:
        events_df: Events dataframe
        people_df: People dataframe
        attendees_df: Attendees dataframe
        date_range: (start_date, end_date) tuple
        event_types: List of event types to include (e.g., ["Free", "Paid"])
        topic_categories: List of topic categories to include
        
    Returns:
        Filtered (events_df, people_df, attendees_df)
    """
    # Convert date_range to pandas Timestamps (timezone-naive)
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    
    # Filter events by date range
    if "Start Date & Time" in events_df.columns:
        events_filtered = events_df[
            (events_df["Start Date & Time"] >= start_date) &
            (events_df["Start Date & Time"] <= end_date)
        ].copy()
    else:
        events_filtered = events_df.copy()
    
    # Filter by event type
    if "Is Free?" in events_filtered.columns:
        if "Both" not in event_types:
            events_filtered = events_filtered[events_filtered["Is Free?"].isin(event_types)]
    
    # Filter by topic category
    if topic_categories and "Topic Category" in events_filtered.columns:
        events_filtered = events_filtered[
            events_filtered["Topic Category"].isin(topic_categories) |
            events_filtered["Topic Category"].isna()
        ]
    
    # Get filtered event IDs
    filtered_event_ids = set(events_filtered["Luma Event ID"].unique())
    
    # Filter attendees by event IDs and date range
    attendees_filtered = attendees_df[
        (attendees_df["Luma Event ID"].isin(filtered_event_ids)) &
        (attendees_df["Event Date"] >= start_date) &
        (attendees_df["Event Date"] <= end_date)
    ].copy()
    
    # Filter people to only those in filtered attendees
    filtered_emails = set(attendees_filtered["Email"].unique())
    people_filtered = people_df[people_df["Email"].isin(filtered_emails)].copy()
    
    return events_filtered, people_filtered, attendees_filtered


def compute_event_cohort_matrix(attendees_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute event-based cohort retention matrix.
    
    Each row = one event (cohort event).
    Each column = kth subsequent event in chronological order.
    Cell (i, j) = % of attendees from event i who also attended the j-th subsequent event.
    
    Only uses the events and attendees passed in (already filtered by date and Free/Paid in apply_filters).
    
    Returns:
        DataFrame with event names as rows, subsequent event offsets as columns, values as retention %
    """
    if attendees_df.empty or events_df.empty:
        return pd.DataFrame()
    
    # Ensure consistent keys
    df_events = events_df.copy()
    df_att = attendees_df.copy()
    
    # Sort events chronologically by Event Date / Start Date & Time
    if "Start Date & Time" in df_events.columns:
        # Use Events table for order
        df_events = df_events.sort_values("Start Date & Time")
    else:
        return pd.DataFrame()
    
    event_ids = df_events["Luma Event ID"].tolist()
    
    # For each event, get set of attendee emails
    df_att["Email"] = df_att["Email"].str.lower()
    event_to_attendees = {
        eid: set(df_att[df_att["Luma Event ID"] == eid]["Email"].dropna().unique())
        for eid in event_ids
    }
    
    # Build retention matrix: rows = cohort event index, cols = subsequent event index
    data = []
    index_labels = []
    
    # We'll also build dynamic column names as we go
    max_offset = 0
    
    for i, cohort_event_id in enumerate(event_ids):
        cohort_attendees = event_to_attendees.get(cohort_event_id, set())
        cohort_size = len(cohort_attendees)
        if cohort_size == 0:
            continue
        
        row = {}
        
        # label row as "Event Name (YYYY-MM-DD)"
        ev_row = df_events[df_events["Luma Event ID"] == cohort_event_id].iloc[0]
        ev_name = str(ev_row.get("Event Name", cohort_event_id))
        ev_date = ev_row.get("Start Date & Time")
        if pd.notna(ev_date):
            if isinstance(ev_date, pd.Timestamp):
                label = f"{ev_name} – {ev_date.date()}"
            else:
                label = f"{ev_name} – {ev_date}"
        else:
            label = ev_name
        
        index_labels.append(label)
        
        # Compare with subsequent events only
        for offset, later_event_id in enumerate(event_ids[i+1:], start=1):
            later_attendees = event_to_attendees.get(later_event_id, set())
            overlap = cohort_attendees & later_attendees
            retention_pct = (len(overlap) / cohort_size * 100.0) if cohort_size > 0 else 0.0
            row[offset] = round(retention_pct, 1)
            if offset > max_offset:
                max_offset = offset
        
        data.append(row)
    
    if not data:
        return pd.DataFrame()
    
    # Create DataFrame with offsets as columns
    matrix = pd.DataFrame(data, index=index_labels)
    
    # Ensure all offset columns exist from 1..max_offset
    for offset in range(1, max_offset + 1):
        if offset not in matrix.columns:
            matrix[offset] = 0.0
    
    # Sort columns by offset
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    
    # Rename columns to "Next Event 1", "Next Event 2", ...
    matrix.columns = [f"Next Event {int(c)}" for c in matrix.columns]
    
    return matrix


def compute_event_retention_table(attendees_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute retention and new attendee percentages for each event.
    
    For each event, calculates:
    - % Retained: % of attendees who attended at least one previous event
    - % New: % of attendees who are new (first time attending any event)
    
    Returns:
        DataFrame with Event Name, Date, Total Attendees, % Retained, % New
    """
    if attendees_df.empty or events_df.empty:
        return pd.DataFrame()
    
    # Sort events chronologically
    df_events = events_df.copy().sort_values("Start Date & Time")
    df_att = attendees_df.copy()
    df_att["Email"] = df_att["Email"].str.lower()
    
    event_ids = df_events["Luma Event ID"].tolist()
    
    # Build a set of all emails seen before each event
    emails_seen_before = set()
    
    retention_data = []
    
    for event_id in event_ids:
        # Get event info
        ev_row = df_events[df_events["Luma Event ID"] == event_id].iloc[0]
        ev_name = str(ev_row.get("Event Name", event_id))
        ev_date = ev_row.get("Start Date & Time")
        
        # Get attendees for this event
        event_attendees = set(df_att[df_att["Luma Event ID"] == event_id]["Email"].dropna().unique())
        total_attendees = len(event_attendees)
        
        if total_attendees == 0:
            continue
        
        # Calculate retained (attended before) vs new
        retained_attendees = event_attendees & emails_seen_before
        new_attendees = event_attendees - emails_seen_before
        
        retained_count = len(retained_attendees)
        new_count = len(new_attendees)
        
        retained_pct = (retained_count / total_attendees * 100.0) if total_attendees > 0 else 0.0
        new_pct = (new_count / total_attendees * 100.0) if total_attendees > 0 else 0.0
        
        # Format date
        if pd.notna(ev_date):
            if isinstance(ev_date, pd.Timestamp):
                date_str = ev_date.date()
            else:
                date_str = ev_date
        else:
            date_str = ""
        
        retention_data.append({
            "Event Name": ev_name,
            "Date": date_str,
            "Total Attendees": total_attendees,
            "Retained Count": retained_count,
            "New Count": new_count,
            "% Retained": round(retained_pct, 1),
            "% New": round(new_pct, 1)
        })
        
        # Update emails_seen_before for next event
        emails_seen_before.update(event_attendees)
    
    return pd.DataFrame(retention_data)


def compute_free_paid_funnel(attendees_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict:
    """
    Compute Free → Paid funnel metrics with detailed event breakdowns.
    
    Returns:
        Dictionary with funnel metrics and detailed event transition data
    """
    # Get first event for each person with event details
    first_events = attendees_df.sort_values("RSVP Time").groupby("Email").first().reset_index()
    first_events = first_events[["Email", "Event Type", "Luma Event ID", "Event Name"]]
    first_events.columns = ["Email", "First Event Type", "First Event ID", "First Event Name"]
    
    # Get all events per person sorted by date
    person_events = attendees_df.sort_values("RSVP Time").groupby("Email").agg({
        "Event Type": list,
        "Luma Event ID": list,
        "Event Name": list
    }).reset_index()
    person_events.columns = ["Email", "All Event Types", "All Event IDs", "All Event Names"]
    
    # Merge
    funnel_data = first_events.merge(person_events, on="Email")
    
    # Determine if person ever attended paid and get first paid event
    def get_first_paid_event(row):
        all_types = row["All Event Types"] if isinstance(row["All Event Types"], list) else []
        all_ids = row["All Event IDs"] if isinstance(row["All Event IDs"], list) else []
        all_names = row["All Event Names"] if isinstance(row["All Event Names"], list) else []
        
        for i, event_type in enumerate(all_types):
            if event_type == "Paid":
                return {
                    "id": all_ids[i] if i < len(all_ids) else "",
                    "name": all_names[i] if i < len(all_names) else ""
                }
        return None
    
    funnel_data["First Paid Event"] = funnel_data.apply(get_first_paid_event, axis=1)
    funnel_data["Ever Attended Paid"] = funnel_data["All Event Types"].apply(
        lambda x: "Paid" in x if isinstance(x, list) else False
    )
    
    # Categorize
    free_first = funnel_data[funnel_data["First Event Type"] == "Free"].copy()
    paid_first = funnel_data[funnel_data["First Event Type"] == "Paid"].copy()
    
    free_to_paid = free_first[free_first["Ever Attended Paid"]].copy()
    free_to_free_only = free_first[~free_first["Ever Attended Paid"]].copy()
    
    # Build Free → Paid event breakdown
    free_to_paid_breakdown = []
    for _, row in free_to_paid.iterrows():
        first_paid = row["First Paid Event"]
        free_to_paid_breakdown.append({
            "First Free Event": row["First Event Name"],
            "First Free Event ID": row["First Event ID"],
            "First Paid Event": first_paid["name"] if first_paid else "Unknown",
            "First Paid Event ID": first_paid["id"] if first_paid else ""
        })
    
    free_to_paid_df = pd.DataFrame(free_to_paid_breakdown) if free_to_paid_breakdown else pd.DataFrame(columns=["First Free Event", "First Free Event ID", "First Paid Event", "First Paid Event ID"])
    
    # Build Paid → Free breakdown
    paid_to_free_breakdown = []
    for _, row in paid_first.iterrows():
        all_types = row["All Event Types"] if isinstance(row["All Event Types"], list) else []
        all_ids = row["All Event IDs"] if isinstance(row["All Event IDs"], list) else []
        all_names = row["All Event Names"] if isinstance(row["All Event Names"], list) else []
        
        # Find first free event (since first event is paid, any free event comes after)
        first_free_after_paid = None
        for i, event_type in enumerate(all_types):
            if event_type == "Free":
                first_free_after_paid = {
                    "id": all_ids[i] if i < len(all_ids) else "",
                    "name": all_names[i] if i < len(all_names) else ""
                }
                break
        
        if first_free_after_paid:
            paid_to_free_breakdown.append({
                "First Paid Event": row["First Event Name"],
                "First Paid Event ID": row["First Event ID"],
                "First Free Event After": first_free_after_paid["name"],
                "First Free Event After ID": first_free_after_paid["id"]
            })
    
    paid_to_free_df = pd.DataFrame(paid_to_free_breakdown) if paid_to_free_breakdown else pd.DataFrame(columns=["First Paid Event", "First Paid Event ID", "First Free Event After", "First Free Event After ID"])
    
    return {
        "free_to_paid": len(free_to_paid),
        "free_to_free_only": len(free_to_free_only),
        "paid_first": len(paid_first),
        "paid_to_free": len(paid_to_free_df),
        "total": len(funnel_data),
        "free_to_paid_pct": (len(free_to_paid) / len(funnel_data) * 100) if len(funnel_data) > 0 else 0,
        "free_to_free_only_pct": (len(free_to_free_only) / len(funnel_data) * 100) if len(funnel_data) > 0 else 0,
        "paid_first_pct": (len(paid_first) / len(funnel_data) * 100) if len(funnel_data) > 0 else 0,
        "paid_to_free_pct": (len(paid_to_free_df) / len(paid_first) * 100) if len(paid_first) > 0 else 0,
        "free_to_paid_breakdown": free_to_paid_df,
        "paid_to_free_breakdown": paid_to_free_df,
        "funnel_data": funnel_data  # Include for time series analysis
    }


def compute_free_to_paid_trend(attendees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Free → Paid conversion rate over time.
    
    Groups people by their first event date (by week) and calculates
    what % of people who started with Free later converted to Paid.
    
    Returns:
        DataFrame with Period (week), Conversion Rate %, Total Free First, Converted Count
    """
    if attendees_df.empty:
        return pd.DataFrame()
    
    # Get first event for each person with date
    first_events = attendees_df.sort_values("RSVP Time").groupby("Email").first().reset_index()
    first_events = first_events[["Email", "Event Type", "Event Date"]]
    first_events.columns = ["Email", "First Event Type", "First Event Date"]
    
    # Get all event types per person
    person_events = attendees_df.groupby("Email")["Event Type"].apply(list).reset_index()
    person_events.columns = ["Email", "All Event Types"]
    
    # Merge
    funnel_data = first_events.merge(person_events, on="Email")
    
    # Determine if person ever attended paid
    funnel_data["Ever Attended Paid"] = funnel_data["All Event Types"].apply(
        lambda x: "Paid" in x if isinstance(x, list) else False
    )
    
    # Filter to only people who started with Free
    free_first = funnel_data[funnel_data["First Event Type"] == "Free"].copy()
    
    if free_first.empty:
        return pd.DataFrame()
    
    # Group by week of first event
    free_first["Period"] = free_first["First Event Date"].dt.to_period("W").dt.start_time
    
    # Calculate conversion rate per week
    trend_data = []
    for period, group in free_first.groupby("Period"):
        total_free_first = len(group)
        converted_count = group["Ever Attended Paid"].sum()
        conversion_rate = (converted_count / total_free_first * 100) if total_free_first > 0 else 0
        
        trend_data.append({
            "Period": period,
            "Conversion Rate %": round(conversion_rate, 1),
            "Total Free First": total_free_first,
            "Converted Count": converted_count
        })
    
    result_df = pd.DataFrame(trend_data)
    result_df = result_df.sort_values("Period")
    
    # Ensure timezone-naive
    if not result_df["Period"].empty:
        sample = result_df["Period"].dropna()
        if len(sample) > 0 and sample.iloc[0].tzinfo is not None:
            result_df["Period"] = result_df["Period"].dt.tz_localize(None)
    
    return result_df


def compute_attendance_frequency(attendees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average gap between events for each person.
    
    Returns:
        DataFrame with Email, Average Gap Days, Frequency Bucket
    """
    # Get events per person sorted by date
    person_events = attendees_df.sort_values("RSVP Time").groupby("Email")["RSVP Time"].apply(list).reset_index()
    
    frequency_data = []
    for _, row in person_events.iterrows():
        email = row["Email"]
        dates = sorted([d for d in row["RSVP Time"] if pd.notna(d)])
        
        if len(dates) < 2:
            continue
        
        # Calculate gaps between consecutive events
        gaps = []
        for i in range(len(dates) - 1):
            gap = (dates[i+1] - dates[i]).days
            gaps.append(gap)
        
        avg_gap = np.mean(gaps) if gaps else 0
        
        # Categorize into week-based ranges
        if avg_gap <= 7:
            bucket = "0-1 week"
        elif avg_gap <= 14:
            bucket = "1-2 weeks"
        elif avg_gap <= 28:
            bucket = "2-4 weeks"
        elif avg_gap <= 56:
            bucket = "4-8 weeks"
        elif avg_gap <= 84:
            bucket = "8-12 weeks"
        elif avg_gap <= 112:
            bucket = "12-16 weeks"
        else:
            bucket = "16+ weeks"
        
        frequency_data.append({
            "Email": email,
            "Average Gap Days": avg_gap,
            "Frequency Bucket": bucket
        })
    
    return pd.DataFrame(frequency_data)


def compute_first_second_gap(attendees_df: pd.DataFrame) -> Dict:
    """
    Compute average gap between first and second event.
    
    Returns:
        Dictionary with metrics
    """
    # Get first two events per person
    person_events = attendees_df.sort_values("RSVP Time").groupby("Email")["RSVP Time"].apply(
        lambda x: x.head(2).tolist()
    ).reset_index()
    
    gaps = []
    for _, row in person_events.iterrows():
        dates = [d for d in row["RSVP Time"] if pd.notna(d)]
        if len(dates) >= 2:
            gap = (dates[1] - dates[0]).days
            gaps.append(gap)
    
    if not gaps:
        return {"avg": 0, "median": 0, "gaps": []}
    
    return {
        "avg": np.mean(gaps),
        "median": np.median(gaps),
        "gaps": gaps
    }


def compute_churn_by_cohort(people_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn rate by cohort.
    Excludes people whose most recent event is less than 3 months (90 days) old.
    
    Returns:
        DataFrame with Cohort Month, Churn Rate, Churned Count, Total Count
    """
    today = pd.Timestamp.now(tz=None)
    
    # Filter to only people whose most recent event is at least 3 months old
    people_with_dates = people_df[pd.notna(people_df["Most Recent Event Date"])].copy()
    people_with_dates["Days Since Last Event"] = (
        today - people_with_dates["Most Recent Event Date"]
    ).dt.days
    
    # Only include people whose last event was at least 90 days ago
    people_eligible = people_with_dates[people_with_dates["Days Since Last Event"] >= 90].copy()
    
    if people_eligible.empty:
        return pd.DataFrame()
    
    churn_by_cohort = people_eligible.groupby("Cohort Month").agg({
        "Churned?": ["sum", "count"]
    }).reset_index()
    
    churn_by_cohort.columns = ["Cohort Month", "Churned Count", "Total Count"]
    churn_by_cohort["Churn Rate %"] = (churn_by_cohort["Churned Count"] / churn_by_cohort["Total Count"] * 100).round(1)
    
    return churn_by_cohort[churn_by_cohort["Cohort Month"].notna()]


def compute_churn_by_event_type(attendees_df: pd.DataFrame, people_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn rate by event type pattern.
    Excludes people whose most recent event is less than 3 months (90 days) old.
    
    Returns:
        DataFrame with Event Type Pattern, Churn Rate
    """
    today = pd.Timestamp.now(tz=None)
    
    # Filter to only people whose most recent event is at least 3 months old
    people_with_dates = people_df[pd.notna(people_df["Most Recent Event Date"])].copy()
    people_with_dates["Days Since Last Event"] = (
        today - people_with_dates["Most Recent Event Date"]
    ).dt.days
    
    # Only include people whose last event was at least 90 days ago
    people_eligible = people_with_dates[people_with_dates["Days Since Last Event"] >= 90].copy()
    
    if people_eligible.empty:
        return pd.DataFrame()
    
    # Get event types per person
    person_types = attendees_df.groupby("Email")["Event Type"].apply(
        lambda x: set(x.unique())
    ).reset_index()
    person_types.columns = ["Email", "Event Types"]
    
    # Categorize
    person_types["Pattern"] = person_types["Event Types"].apply(
        lambda x: "Only Free" if x == {"Free"} else "At least 1 Paid"
    )
    
    # Merge with eligible people to get churn status
    merged = person_types.merge(people_eligible[["Email", "Churned?"]], on="Email", how="inner")
    
    churn_by_type = merged.groupby("Pattern").agg({
        "Churned?": ["sum", "count"]
    }).reset_index()
    
    churn_by_type.columns = ["Pattern", "Churned Count", "Total Count"]
    churn_by_type["Churn Rate %"] = (churn_by_type["Churned Count"] / churn_by_type["Total Count"] * 100).round(1)
    
    return churn_by_type


def compute_new_attendees(people_df: pd.DataFrame, period: str = "week") -> pd.DataFrame:
    """
    Compute new attendees per week or month.
    
    Args:
        people_df: People dataframe
        period: "week" or "month"
        
    Returns:
        DataFrame with Period, New Attendees
    """
    if "First Event Date" not in people_df.columns:
        return pd.DataFrame()
    
    people_with_dates = people_df[people_df["First Event Date"].notna()].copy()
    
    if period == "week":
        # Convert to period, then get start date of period (first day of week)
        people_with_dates["Period"] = people_with_dates["First Event Date"].dt.to_period("W").dt.start_time
    else:
        # Convert to period, then get start date of period (first day of month)
        people_with_dates["Period"] = people_with_dates["First Event Date"].dt.to_period("M").dt.start_time
    
    new_attendees = people_with_dates.groupby("Period").size().reset_index()
    new_attendees.columns = ["Period", "New Attendees"]
    # Period is already a datetime, just ensure it's timezone-naive
    # Check if timezone-aware by looking at actual values
    if not new_attendees["Period"].empty:
        sample = new_attendees["Period"].dropna()
        if len(sample) > 0 and sample.iloc[0].tzinfo is not None:
            new_attendees["Period"] = new_attendees["Period"].dt.tz_localize(None)
    new_attendees = new_attendees.sort_values("Period")
    
    return new_attendees


def compute_event_churn(attendees_df: pd.DataFrame, events_df: pd.DataFrame, min_attendees: int = 10) -> pd.DataFrame:
    """
    Compute churn rate for each event.
    Excludes events that are less than 3 months (90 days) old, as churn cannot be determined yet.
    
    Returns:
        DataFrame with Event Name, Date, Total Attendees, Churn Rate
    """
    today = pd.Timestamp.now(tz=None)
    event_churn_data = []
    
    for event_id in attendees_df["Luma Event ID"].unique():
        event_attendees = attendees_df[attendees_df["Luma Event ID"] == event_id]
        
        if len(event_attendees) < min_attendees:
            continue
        
        event_date = event_attendees["Event Date"].iloc[0]
        
        # Exclude events less than 3 months (90 days) old
        days_since_event = (today - event_date).days if pd.notna(event_date) else None
        if days_since_event is None or days_since_event < 90:
            continue
        
        attendee_emails = set(event_attendees["Email"].unique())
        
        # Find attendees who never appear in later events
        later_events = attendees_df[attendees_df["Event Date"] > event_date]
        later_attendee_emails = set(later_events["Email"].unique())
        
        never_returned = attendee_emails - later_attendee_emails
        churn_rate = (len(never_returned) / len(attendee_emails) * 100) if attendee_emails else 0
        
        # Get event name
        event_info = events_df[events_df["Luma Event ID"] == event_id]
        event_name = event_info["Event Name"].iloc[0] if not event_info.empty else "Unknown"
        
        event_churn_data.append({
            "Event Name": event_name,
            "Date": event_date,
            "Total Attendees": len(attendee_emails),
            "Churn Rate %": round(churn_rate, 1)
        })
    
    result_df = pd.DataFrame(event_churn_data)
    return result_df.sort_values("Churn Rate %", ascending=False).head(10)


def compute_monthly_churn_trend(people_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn rate trend over time by cohort month.
    
    For each month (cohort), calculates the churn rate of people whose first event
    was in that month. Only includes cohorts that are old enough (at least 3 months
    since their last event) to accurately measure churn. This shows how retention
    varies across different cohorts over time.
    
    Returns:
        DataFrame with Period (month), Churn Rate %, Total People, Churned Count
    """
    if people_df.empty:
        return pd.DataFrame()
    
    today = pd.Timestamp.now(tz=None)
    
    # Filter to people with cohort month and most recent event date
    people_with_data = people_df[
        people_df["Cohort Month"].notna() & 
        people_df["Most Recent Event Date"].notna()
    ].copy()
    
    if people_with_data.empty:
        return pd.DataFrame()
    
    # Calculate days since last event for each person
    people_with_data["Days Since Last Event"] = (
        today - people_with_data["Most Recent Event Date"]
    ).dt.days
    
    # Only include people whose last event is at least 90 days ago (eligible for churn measurement)
    people_eligible = people_with_data[people_with_data["Days Since Last Event"] >= 90].copy()
    
    if people_eligible.empty:
        return pd.DataFrame()
    
    # Group by cohort month
    trend_data = []
    
    for cohort_month, group in people_eligible.groupby("Cohort Month"):
        if pd.isna(cohort_month):
            continue
        
        total_people = len(group)
        churned_count = group["Churned?"].sum() if "Churned?" in group.columns else 0
        churn_rate = (churned_count / total_people * 100) if total_people > 0 else 0
        
        # Convert cohort month to datetime if it's a period
        if isinstance(cohort_month, pd.Period):
            period_start = cohort_month.start_time
        else:
            period_start = pd.to_datetime(cohort_month)
        
        trend_data.append({
            "Period": period_start,
            "Churn Rate %": round(churn_rate, 1),
            "Total People": total_people,
            "Churned Count": churned_count
        })
    
    result_df = pd.DataFrame(trend_data)
    result_df = result_df.sort_values("Period")
    
    # Ensure timezone-naive
    if not result_df["Period"].empty:
        sample = result_df["Period"].dropna()
        if len(sample) > 0 and sample.iloc[0].tzinfo is not None:
            result_df["Period"] = result_df["Period"].dt.tz_localize(None)
    
    return result_df


def compute_paid_event_behavior(attendees_df: pd.DataFrame) -> Dict:
    """
    Compute behavior after first paid event.
    
    Returns:
        Dictionary with metrics
    """
    # Get people who attended at least one paid event
    paid_attendees = attendees_df[attendees_df["Event Type"] == "Paid"]["Email"].unique()
    
    if len(paid_attendees) == 0:
        return {"paid_to_paid": 0, "paid_to_free_only": 0, "total": 0}
    
    # For each person, find their first paid event
    paid_events = attendees_df[attendees_df["Event Type"] == "Paid"].sort_values("RSVP Time")
    first_paid = paid_events.groupby("Email").first().reset_index()
    first_paid = first_paid[["Email", "RSVP Time"]]
    first_paid.columns = ["Email", "First Paid Date"]
    
    # Get all events after first paid
    behavior_data = []
    for _, row in first_paid.iterrows():
        email = row["Email"]
        first_paid_date = row["First Paid Date"]
        
        later_events = attendees_df[
            (attendees_df["Email"] == email) &
            (attendees_df["RSVP Time"] > first_paid_date)
        ]
        
        if len(later_events) == 0:
            continue
        
        later_types = set(later_events["Event Type"].unique())
        has_paid_after = "Paid" in later_types
        has_free_after = "Free" in later_types
        
        if has_paid_after:
            behavior_data.append("Paid → Another Paid")
        elif has_free_after:
            behavior_data.append("Paid → Only Free afterwards")
    
    behavior_counts = pd.Series(behavior_data).value_counts().to_dict()
    
    return {
        "paid_to_paid": behavior_counts.get("Paid → Another Paid", 0),
        "paid_to_free_only": behavior_counts.get("Paid → Only Free afterwards", 0),
        "total": len(behavior_data)
    }


def compute_flakes_analysis(attendees_df: pd.DataFrame) -> Dict:
    """
    Compute flake and churn analysis.
    Uses 3-month (90 day) churn definition: churned if no event in last 90 days.
    
    Returns:
        Dictionary with flake and churn metrics
    """
    total_registrations = len(attendees_df)
    total_flakes = attendees_df["Flaked?"].sum()
    flake_rate = (total_flakes / total_registrations * 100) if total_registrations > 0 else 0
    
    # Flake rate by event type
    flake_by_type = attendees_df.groupby("Event Type").agg({
        "Flaked?": ["sum", "count"]
    }).reset_index()
    flake_by_type.columns = ["Event Type", "Flakes", "Total"]
    flake_by_type["Flake Rate %"] = (flake_by_type["Flakes"] / flake_by_type["Total"] * 100).round(1)
    
    # People who flaked - check if they churned (90 days since last event)
    people_who_flaked = set(attendees_df[attendees_df["Flaked?"]]["Email"].unique())
    
    if len(people_who_flaked) == 0:
        return {
            "flake_rate": flake_rate,
            "flake_by_type": flake_by_type,
            "flaked_and_churned": 0,
            "flaked_and_not_churned": 0,
            "total_flakers": 0
        }
    
    today = pd.Timestamp.now(tz=None)
    churned_after_flake = 0
    not_churned_after_flake = 0
    
    for email in people_who_flaked:
        person_events = attendees_df[attendees_df["Email"] == email].sort_values("Event Date")
        
        # Get their most recent event date (any event, not just checked in)
        if person_events.empty:
            continue
        
        most_recent_event_date = person_events["Event Date"].max()
        
        # Check if churned: no event in last 90 days
        days_since_last = (today - most_recent_event_date).days if pd.notna(most_recent_event_date) else None
        
        if days_since_last is not None and days_since_last > 90:
            churned_after_flake += 1
        else:
            not_churned_after_flake += 1
    
    return {
        "flake_rate": flake_rate,
        "flake_by_type": flake_by_type,
        "flaked_and_churned": churned_after_flake,
        "flaked_and_not_churned": not_churned_after_flake,
        "total_flakers": len(people_who_flaked)
    }


def main():
    """Main Streamlit app."""
    st.title("📊 Amplifier Analytics – Cohorts & Funnels")
    
    # Load data
    with st.spinner("Loading data..."):
        events_df, people_df, attendees_df = load_data()
    
    # Validate data was loaded successfully
    if events_df.empty or people_df.empty or attendees_df.empty:
        st.error("❌ Error: One or more data files are empty. Please ensure the CSV files contain data.")
        st.stop()
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    if "Event Date" in attendees_df.columns and not attendees_df["Event Date"].isna().all():
        min_date = attendees_df["Event Date"].min().date()
        max_date = attendees_df["Event Date"].max().date()
        
        date_range = st.sidebar.date_input(
            "Event Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        else:
            date_range = (pd.Timestamp(min_date), pd.Timestamp(max_date))
    else:
        date_range = (pd.Timestamp("2020-01-01"), pd.Timestamp.now())
    
    # Event type filter
    event_types = st.sidebar.multiselect(
        "Event Type",
        options=["Free", "Paid", "Both"],
        default=["Both"]
    )
    
    if "Both" in event_types:
        event_types = ["Free", "Paid"]
    
    # Topic category filter
    if "Topic Category" in events_df.columns:
        topic_options = ["All"] + sorted(events_df["Topic Category"].dropna().unique().tolist())
        selected_topics = st.sidebar.multiselect(
            "Topic Category",
            options=topic_options,
            default=["All"]
        )
        
        if "All" in selected_topics or len(selected_topics) == 0:
            topic_categories = []
        else:
            topic_categories = selected_topics
    else:
        topic_categories = []
    
    # Apply filters
    events_filt, people_filt, attendees_filt = apply_filters(
        events_df, people_df, attendees_df, date_range, event_types, topic_categories
    )
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Cohorts & Funnels", "Attendance Patterns", "Churn & Retention", "Flakes & Churn"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(events_filt))
        
        with col2:
            st.metric("Total People", len(people_filt))
        
        with col3:
            st.metric("Total Attendee Records", len(attendees_filt))
        
        with col4:
            total_checked_in = attendees_filt["Checked In"].sum() if "Checked In" in attendees_filt.columns else 0
            st.metric("Total Check-ins", total_checked_in)
        
        # Events table
        st.subheader("All Events")
        st.caption("All events ordered by event date (ascending).")
        
        if not events_filt.empty and "Start Date & Time" in events_filt.columns:
            # Sort by event date ascending
            events_display = events_filt.sort_values("Start Date & Time", ascending=True).copy()
            # Display all columns from events CSV
            st.dataframe(events_display, use_container_width=True, hide_index=True)
        else:
            st.info("No events data available.")
        
        # New attendees over time
        st.subheader("New Attendees Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weekly_new = compute_new_attendees(people_filt, "week")
            if not weekly_new.empty:
                fig = px.line(weekly_new, x="Period", y="New Attendees", title="Weekly New Attendees")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_new = compute_new_attendees(people_filt, "month")
            if not monthly_new.empty:
                fig = px.line(monthly_new, x="Period", y="New Attendees", title="Monthly New Attendees")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Cohorts & Funnels
    with tab2:
        st.header("Cohort Analysis")
        
        # Event retention table
        st.subheader("Event Retention & New Attendees")
        st.caption("For each event: % of attendees who attended previous events (Retained) vs % who are new. Sort by any column.")
        
        retention_table = compute_event_retention_table(attendees_filt, events_filt)
        
        if not retention_table.empty:
            # Format percentage columns for display
            display_table = retention_table.copy()
            display_table["% Retained"] = display_table["% Retained"].apply(lambda x: f"{x:.1f}%")
            display_table["% New"] = display_table["% New"].apply(lambda x: f"{x:.1f}%")
            
            # Display sortable table
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No retention data available for the current filters.")
        
        # Free → Paid Funnel
        st.subheader("Free → Paid Funnel")
        st.caption("Analyzes conversion from free events to paid events.")
        
        # Add definitions
        with st.expander("📊 Metric Definitions", expanded=False):
            st.markdown("""
            **Metric Definitions:**
            - **Free → Paid**: People whose first event was Free and later attended at least one Paid event (% of total people)
            - **Free → Free Only**: People whose first event was Free and only attended Free events (% of total people)
            - **Paid First**: People whose first event was Paid (% of total people)
            - **Paid → Free**: People who started with Paid and later attended at least one Free event (% of people who started with Paid)
            
            Note: The first three percentages add up to 100% of total people. "Paid → Free" is a conditional percentage showing what % of paid-first people later attended free events.
            """)
        
        funnel_metrics = compute_free_paid_funnel(attendees_filt, events_filt)
        
        total = funnel_metrics['total']
        paid_to_free_pct_of_total = (funnel_metrics['paid_to_free'] / total * 100) if total > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Free → Paid", f"{funnel_metrics['free_to_paid']}", delta=f"{funnel_metrics['free_to_paid_pct']:.1f}% of total")
        with col2:
            st.metric("Free → Free Only", f"{funnel_metrics['free_to_free_only']}", delta=f"{funnel_metrics['free_to_free_only_pct']:.1f}% of total")
        with col3:
            st.metric("Paid First", f"{funnel_metrics['paid_first']}", delta=f"{funnel_metrics['paid_first_pct']:.1f}% of total")
        with col4:
            st.metric("Paid → Free", f"{funnel_metrics['paid_to_free']}", 
                     delta=f"{funnel_metrics['paid_to_free_pct']:.1f}% of Paid First ({paid_to_free_pct_of_total:.1f}% of total)")
        
        # Funnel visualization
        funnel_data = pd.DataFrame({
            "Stage": ["First Event = Free", "Free → Paid", "Free → Free Only", "First Event = Paid"],
            "Count": [
                funnel_metrics['free_to_paid'] + funnel_metrics['free_to_free_only'],
                funnel_metrics['free_to_paid'],
                funnel_metrics['free_to_free_only'],
                funnel_metrics['paid_first']
            ]
        })
        
        fig = px.bar(funnel_data, x="Stage", y="Count", title="Funnel: First Event Type & Conversion")
        st.plotly_chart(fig, use_container_width=True)
        
        # Verification: Show that percentages add up
        total_pct = funnel_metrics['free_to_paid_pct'] + funnel_metrics['free_to_free_only_pct'] + funnel_metrics['paid_first_pct']
        st.caption(f"✓ Verification: Free → Paid ({funnel_metrics['free_to_paid_pct']:.1f}%) + Free → Free Only ({funnel_metrics['free_to_free_only_pct']:.1f}%) + Paid First ({funnel_metrics['paid_first_pct']:.1f}%) = {total_pct:.1f}% of total")
        
        # Free → Paid Conversion Trend Over Time
        st.subheader("Free → Paid Conversion Trend Over Time")
        st.caption("Shows how the Free → Paid conversion rate has changed over time, grouped by week of first event.")
        
        conversion_trend = compute_free_to_paid_trend(attendees_filt)
        if not conversion_trend.empty:
            fig = px.line(
                conversion_trend,
                x="Period",
                y="Conversion Rate %",
                title="Free → Paid Conversion Rate Over Time",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Week of First Event",
                yaxis_title="Conversion Rate (%)",
                hovermode='x unified'
            )
            fig.update_traces(
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Conversion Rate: %{y:.1f}%<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Conversion Rate", f"{conversion_trend['Conversion Rate %'].mean():.1f}%")
            with col2:
                st.metric("Latest Conversion Rate", f"{conversion_trend['Conversion Rate %'].iloc[-1]:.1f}%")
            with col3:
                if len(conversion_trend) > 1:
                    first_rate = conversion_trend['Conversion Rate %'].iloc[0]
                    latest_rate = conversion_trend['Conversion Rate %'].iloc[-1]
                    change = latest_rate - first_rate
                    st.metric("Change (First → Latest)", f"{change:+.1f}%")
        else:
            st.info("No data available for conversion trend analysis.")
        
        # Free → Paid Event Breakdown
        if not funnel_metrics['free_to_paid_breakdown'].empty:
            st.subheader("Free → Paid: Event Breakdown")
            st.caption("Which free events led to paid conversions, and which paid events they converted to.")
            
            free_to_paid_df = funnel_metrics['free_to_paid_breakdown']
            
            # Group by first free event
            free_event_counts = free_to_paid_df.groupby("First Free Event").size().reset_index()
            free_event_counts.columns = ["First Free Event", "Count"]
            free_event_counts = free_event_counts.sort_values("Count", ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Free Events Leading to Paid Conversion:**")
                fig = px.bar(
                    free_event_counts.head(10),
                    x="Count",
                    y="First Free Event",
                    orientation='h',
                    title="Top Free Events → Paid Conversions"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Group by first paid event
            paid_event_counts = free_to_paid_df.groupby("First Paid Event").size().reset_index()
            paid_event_counts.columns = ["First Paid Event", "Count"]
            paid_event_counts = paid_event_counts.sort_values("Count", ascending=False)
            
            with col2:
                st.write("**Paid Events They Converted To:**")
                fig = px.bar(
                    paid_event_counts.head(10),
                    x="Count",
                    y="First Paid Event",
                    orientation='h',
                    title="Paid Events Converted To"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.write("**Detailed Free → Paid Transitions:**")
            st.dataframe(free_to_paid_df, use_container_width=True, hide_index=True)
        
        # Paid → Free Event Breakdown
        st.subheader("Paid → Free: Event Breakdown")
        st.caption("Which paid events led to free event attendance, and which free events they attended.")
        
        paid_to_free_df = funnel_metrics['paid_to_free_breakdown']
        
        if not paid_to_free_df.empty:
            # Group by first paid event
            paid_first_counts = paid_to_free_df.groupby("First Paid Event").size().reset_index()
            paid_first_counts.columns = ["First Paid Event", "Count"]
            paid_first_counts = paid_first_counts.sort_values("Count", ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Paid Events Leading to Free Attendance:**")
                fig = px.bar(
                    paid_first_counts.head(10),
                    x="Count",
                    y="First Paid Event",
                    orientation='h',
                    title="Top Paid Events → Free Attendance"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Group by first free event after paid
            free_after_counts = paid_to_free_df.groupby("First Free Event After").size().reset_index()
            free_after_counts.columns = ["First Free Event After", "Count"]
            free_after_counts = free_after_counts.sort_values("Count", ascending=False)
            
            with col2:
                st.write("**Free Events They Attended After Paid:**")
                fig = px.bar(
                    free_after_counts.head(10),
                    x="Count",
                    y="First Free Event After",
                    orientation='h',
                    title="Free Events Attended After Paid"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.write("**Detailed Paid → Free Transitions:**")
            st.dataframe(paid_to_free_df, use_container_width=True, hide_index=True)
        else:
            st.info("No Paid → Free transitions found. People who started with paid events haven't attended any free events yet.")
    
    # Tab 3: Attendance Patterns
    with tab3:
        st.header("Attendance Patterns")
        
        # Frequency analysis
        st.subheader("Attendance Frequency")
        st.caption("How often people attend events based on average gap between their events.")
        
        frequency_df = compute_attendance_frequency(attendees_filt)
        
        if not frequency_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                bucket_counts = frequency_df["Frequency Bucket"].value_counts()
                # Define bucket order for proper sorting
                bucket_order = ["0-1 week", "1-2 weeks", "2-4 weeks", "4-8 weeks", 
                               "8-12 weeks", "12-16 weeks", "16+ weeks"]
                # Reindex to maintain chronological order
                bucket_counts = bucket_counts.reindex([b for b in bucket_order if b in bucket_counts.index])
                
                fig = px.bar(
                    x=bucket_counts.index,
                    y=bucket_counts.values,
                    title="People Count by Frequency Bucket (Week Ranges)",
                    labels={"x": "Frequency Bucket", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create week-based bins (7-day intervals)
                max_days = frequency_df["Average Gap Days"].max() if not frequency_df.empty else 200
                # Create bins in 1-week (7-day) intervals
                n_weeks = int(np.ceil(max_days / 7)) + 1
                fig = px.histogram(
                    frequency_df,
                    x="Average Gap Days",
                    nbins=n_weeks,
                    title="Distribution of Average Days Between Events (Week-based bins)"
                )
                # Update x-axis to show week markers
                fig.update_xaxes(
                    tickmode='linear',
                    tick0=0,
                    dtick=7,
                    title="Average Gap (days)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # First to second event gap
        st.subheader("First → Second Event Gap")
        st.caption("Time between a person's first and second event registration.")
        
        gap_metrics = compute_first_second_gap(attendees_filt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Gap", f"{gap_metrics['avg']:.1f} days")
        with col2:
            st.metric("Median Gap", f"{gap_metrics['median']:.1f} days")
        
        if gap_metrics['gaps']:
            # Create week-based bins (7-day intervals)
            max_gap = max(gap_metrics['gaps']) if gap_metrics['gaps'] else 200
            n_weeks = int(np.ceil(max_gap / 7)) + 1
            fig = px.histogram(
                x=gap_metrics['gaps'],
                nbins=n_weeks,
                title="Distribution of Days Between First and Second Event (Week-based bins)"
            )
            # Update x-axis to show week markers
            fig.update_xaxes(
                tickmode='linear',
                tick0=0,
                dtick=7,
                title="Days Between Events"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Churn & Retention
    with tab4:
        st.header("Churn & Retention")
        
        # Churn definition
        st.info("**Churn Definition:** A person is considered 'churned' if they have not attended any event in the last 3 months (90 days) from their most recent event date.")
        
        # Overall churn (only include people whose last event is at least 3 months old)
        today = pd.Timestamp.now(tz=None)
        people_with_dates = people_filt[pd.notna(people_filt["Most Recent Event Date"])].copy()
        people_with_dates["Days Since Last Event"] = (
            today - people_with_dates["Most Recent Event Date"]
        ).dt.days
        
        # Only include people whose last event was at least 90 days ago
        people_eligible = people_with_dates[people_with_dates["Days Since Last Event"] >= 90].copy()
        
        total_people = len(people_eligible)
        churned_count = people_eligible["Churned?"].sum() if "Churned?" in people_eligible.columns else 0
        churn_rate = (churned_count / total_people * 100) if total_people > 0 else 0
        
        excluded_count = len(people_filt) - len(people_eligible)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total People (Eligible)", total_people, 
                     help="Only includes people whose most recent event is at least 3 months old")
        with col2:
            st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"-{churned_count} churned")
        with col3:
            st.metric("Excluded (Recent Events)", excluded_count,
                     help="People whose most recent event is less than 3 months old")
        
        # Monthly churn rate trend
        st.subheader("Churn Rate Trend Over Time (By Cohort)")
        st.caption("For each cohort month (first event month), shows the churn rate of people who started in that month. Only includes people whose last event is at least 3 months old to accurately measure churn. This shows how retention varies across different cohorts.")
        
        monthly_churn_trend = compute_monthly_churn_trend(people_filt)
        if not monthly_churn_trend.empty:
            fig = px.line(
                monthly_churn_trend,
                x="Period",
                y="Churn Rate %",
                title="Churn Rate Over Time (Monthly)",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Churn Rate (%)",
                hovermode='x unified'
            )
            fig.update_traces(
                hovertemplate="<b>%{x|%Y-%m}</b><br>Churn Rate: %{y:.1f}%<br>Total People: %{customdata[0]}<extra></extra>",
                customdata=monthly_churn_trend[["Total People"]].values
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for monthly churn trend (need events at least 3 months old).")
        
        # Churn by cohort
        st.subheader("Churn Rate by Cohort")
        st.caption("Churn rate for each cohort month.")
        
        churn_by_cohort = compute_churn_by_cohort(people_filt)
        
        if not churn_by_cohort.empty:
            fig = px.bar(
                churn_by_cohort,
                x="Cohort Month",
                y="Churn Rate %",
                title="Churn Rate by Cohort Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn by event type pattern
        st.subheader("Churn Rate by Event Type Pattern")
        st.caption("Churn rate for people who only attended free events vs. those who attended at least one paid event.")
        
        churn_by_type = compute_churn_by_event_type(attendees_filt, people_filt)
        
        if not churn_by_type.empty:
            fig = px.bar(
                churn_by_type,
                x="Pattern",
                y="Churn Rate %",
                title="Churn Rate by Event Type Pattern"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Event-level churn
        st.subheader("Top 10 Events by Churn Rate")
        st.caption("Events with highest % of attendees who never returned to any later event. Only includes events that are at least 3 months old (90 days), as churn cannot be determined for more recent events.")
        
        event_churn = compute_event_churn(attendees_filt, events_filt)
        
        if not event_churn.empty:
            st.dataframe(event_churn, use_container_width=True)
            
            fig = px.bar(
                event_churn,
                x="Event Name",
                y="Churn Rate %",
                title="Top 10 Events by Churn Rate"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Flakes & Churn
    with tab5:
        st.header("Flakes & Churn")
        st.info("**Churn Definition:** A person is considered 'churned' if they have not attended any event in the last 3 months (90 days) from their most recent event date.")
        
        flakes_analysis = compute_flakes_analysis(attendees_filt)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Flake Rate", f"{flakes_analysis['flake_rate']:.1f}%")
        with col2:
            st.metric("People Who Flaked", flakes_analysis['total_flakers'])
        with col3:
            churn_rate_after_flake = (flakes_analysis['flaked_and_churned'] / flakes_analysis['total_flakers'] * 100) if flakes_analysis['total_flakers'] > 0 else 0
            st.metric("Churn Rate (After Flake)", f"{churn_rate_after_flake:.1f}%")
        
        # Flake rate by event type
        st.subheader("Flake Rate by Event Type")
        st.caption("Percentage of registrations that resulted in no check-in, by event type.")
        
        if not flakes_analysis['flake_by_type'].empty:
            fig = px.bar(
                flakes_analysis['flake_by_type'],
                x="Event Type",
                y="Flake Rate %",
                title="Flake Rate by Event Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Flaked and churned
        st.subheader("Flakes & Churn Analysis")
        st.caption("Of people who flaked at least once, how many churned (no event in last 90 days)?")
        
        churn_data = pd.DataFrame({
            "Status": ["Flaked & Churned", "Flaked & Not Churned"],
            "Count": [
                flakes_analysis['flaked_and_churned'],
                flakes_analysis['flaked_and_not_churned']
            ]
        })
        
        fig = px.bar(churn_data, x="Status", y="Count", title="Flakes & Churn Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Paid event behavior
        st.subheader("Paid Event Behavior")
        st.caption("Behavior of people after their first paid event.")
        
        paid_behavior = compute_paid_event_behavior(attendees_filt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Paid → Another Paid", paid_behavior['paid_to_paid'])
        with col2:
            st.metric("Paid → Only Free Afterwards", paid_behavior['paid_to_free_only'])
        
        if paid_behavior['total'] > 0:
            behavior_data = pd.DataFrame({
                "Behavior": ["Paid → Another Paid", "Paid → Only Free afterwards"],
                "Count": [paid_behavior['paid_to_paid'], paid_behavior['paid_to_free_only']]
            })
            fig = px.bar(behavior_data, x="Behavior", y="Count", title="Behavior After First Paid Event")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
