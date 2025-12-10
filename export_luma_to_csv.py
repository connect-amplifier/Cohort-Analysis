#!/usr/bin/env python3
"""
Export Luma event and guest data to CSV files for Airtable import.

This script fetches all events and their guests from the Luma API,
processes the data, and exports three CSV files:
- events.csv: One row per event
- people.csv: One row per unique person (by email)
- attendees.csv: One row per (person, event) combination
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import requests
import pandas as pd


# Hardcoded API key
LUMA_API_KEY = "secret-5V7JJYnjaCVHWvPFnyd8E5Lcr"

# API endpoints
EVENTS_ENDPOINT = "https://public-api.luma.com/v1/calendar/list-events"
GUESTS_ENDPOINT = "https://public-api.luma.com/v1/event/get-guests"

# Rate limiting delay between requests (seconds)
REQUEST_DELAY = 0.2


def fetch_all_events(api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch all events from the Luma API, handling pagination.
    
    Args:
        api_key: Luma API key
        
    Returns:
        List of event dictionaries (extracted from entries)
    """
    all_events = []
    cursor = None
    page = 1
    
    headers = {
        "x-luma-api-key": api_key,
        "Accept": "application/json"
    }
    
    while True:
        params = {}
        if cursor:
            params["cursor"] = cursor
        
        try:
            print(f"Fetching events page {page}...")
            response = requests.get(EVENTS_ENDPOINT, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # API returns: {"entries": [{"api_id": "...", "event": {...}, "tags": [...]}], "has_more": bool}
            if isinstance(data, dict):
                entries = data.get("entries", [])
                # Extract the actual event data from each entry
                events = [entry.get("event", {}) for entry in entries if isinstance(entry, dict) and "event" in entry]
                has_more = data.get("has_more", False)
                cursor = data.get("next_cursor") or data.get("cursor")
            else:
                events = []
                has_more = False
                cursor = None
            
            if events:
                all_events.extend(events)
                print(f"  Found {len(events)} events on this page")
            
            # If no more pages, break
            if not has_more:
                break
            
            # If has_more but no cursor, we can't paginate further (API limitation)
            if has_more and not cursor:
                print("  Warning: API indicates more events available but no cursor provided. Stopping pagination.")
                break
            
            if cursor:
                page += 1
                time.sleep(REQUEST_DELAY)
            else:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching events page {page}: {e}")
            break
    
    print(f"Fetched {len(all_events)} total events")
    return all_events


def fetch_all_guests_for_event(api_key: str, event_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all guests for a specific event, handling pagination.
    Includes ALL sign-ups (approved, declined, etc.) - everyone who registered counts as an attendee.
    Declined/cancelled guests will be marked as "flaked" in the output.
    
    Fetches guests with multiple approval_statuses to ensure we get everyone who registered.
    
    Args:
        api_key: Luma API key
        event_id: Luma event ID
        
    Returns:
        List of guest dictionaries (extracted from entries, includes all sign-ups)
    """
    all_guests = []
    max_retries = 3
    
    headers = {
        "x-luma-api-key": api_key,
        "Accept": "application/json"
    }
    
    # Fetch guests for each approval status we care about
    # Exclude "invited" - those never actually registered
    approval_statuses = ["approved", "declined", "pending_approval", "waitlist", "session"]
    
    for approval_status in approval_statuses:
        cursor = None
        page = 1
        
        params = {
            "event_id": event_id,
            "approval_status": approval_status,  # Fetch each status explicitly
            "pagination_limit": 100,  # Get more results per page (reduce API calls)
            "sort_column": "registered_at",  # Sort by registration for consistency
            "sort_direction": "desc"  # Most recent first
        }
        
        while True:
            # Update params for pagination (don't overwrite the base params)
            current_params = params.copy()
            if cursor:
                current_params["pagination_cursor"] = cursor  # Use correct parameter name from API docs
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = requests.get(GUESTS_ENDPOINT, headers=headers, params=current_params)
                    
                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        wait_time = (2 ** retry_count) * REQUEST_DELAY * 10  # Exponential backoff
                        print(f"  Rate limited, waiting {wait_time:.1f}s before retry {retry_count + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # API returns: {"entries": [{"api_id": "...", "guest": {...}}], "has_more": bool, "next_cursor": "..."}
                    if isinstance(data, dict):
                        entries = data.get("entries", [])
                        # Extract the actual guest data from each entry
                        # Since we're fetching by approval_status, all returned guests should be included
                        guests = []
                        for entry in entries:
                            if isinstance(entry, dict) and "guest" in entry:
                                guest = entry.get("guest", {})
                                # Include all guests returned (they all have the status we requested)
                                guests.append(guest)
                        
                        has_more = data.get("has_more", False)
                        cursor = data.get("next_cursor")  # Use next_cursor from API response
                    else:
                        guests = []
                        has_more = False
                        cursor = None
                    
                    if guests:
                        all_guests.extend(guests)
                    
                    # Break out of retry loop on success
                    break
                    
                except requests.exceptions.RequestException as e:
                    if retry_count < max_retries - 1:
                        wait_time = (2 ** retry_count) * REQUEST_DELAY * 10
                        print(f"  Error fetching guests (attempt {retry_count + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"  Warning: Error fetching guests for event {event_id} (status={approval_status}) after {max_retries} attempts: {e}")
                        break  # Continue to next status instead of returning
        
            if not has_more and not cursor:
                break
            
            if cursor:
                page += 1
                time.sleep(REQUEST_DELAY)
            else:
                break
        
        # Small delay between different approval status requests
        time.sleep(REQUEST_DELAY)
    
    # Deduplicate guests by their ID (in case there are any overlaps)
    seen_ids = set()
    unique_guests = []
    for guest in all_guests:
        guest_id = guest.get("id") or guest.get("api_id")
        if guest_id and guest_id not in seen_ids:
            seen_ids.add(guest_id)
            unique_guests.append(guest)
    
    return unique_guests


def parse_price(event: Dict[str, Any], guests: Optional[List[Dict[str, Any]]] = None) -> float:
    """
    Extract price from event or guest tickets, defaulting to 0 if not present.
    Uses max ticket value paid (only counts tickets with amount > 0).
    
    Args:
        event: Event dictionary
        guests: Optional list of guests for this event (to check ticket prices)
        
    Returns:
        Price as float (defaults to 0.0)
    """
    # First try to get price from event directly
    price = event.get("ticket_price") or event.get("price") or event.get("cost")
    
    # If not found, check guest tickets for the highest PAID price (amount > 0)
    if price is None and guests:
        max_price = 0.0
        for guest in guests:
            ticket = guest.get("event_ticket", {})
            if isinstance(ticket, dict):
                ticket_price = ticket.get("amount", 0)
                try:
                    ticket_price = float(ticket_price)
                    # Only count paid tickets (amount > 0)
                    if ticket_price > 0:
                        max_price = max(max_price, ticket_price)
                except (ValueError, TypeError):
                    pass
        if max_price > 0:
            price = max_price
    
    try:
        return float(price) if price is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def parse_datetime(dt_str: Optional[str]) -> Optional[str]:
    """Parse datetime string to ISO 8601 format."""
    if not dt_str:
        return None
    try:
        # Try parsing with pandas (handles many formats)
        dt = pd.to_datetime(dt_str)
        return dt.isoformat()
    except:
        return dt_str  # Return as-is if parsing fails


def build_events_df(events: List[Dict[str, Any]], guests_by_event: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Build the events DataFrame with required columns.
    
    Args:
        events: List of event dictionaries
        guests_by_event: Dictionary mapping event_id to list of guests
        
    Returns:
        DataFrame with event data
    """
    rows = []
    
    for event in events:
        event_id = event.get("id", "") or event.get("api_id", "")
        guests = guests_by_event.get(event_id, [])
        price = parse_price(event, guests)
        is_free = "Free" if price == 0 else "Paid"
        
        # Count total attendees for this event
        total_attendees = len(guests)
        
        # Extract location from geo_address_json.full_address if available
        geo_address = event.get("geo_address_json", {})
        if isinstance(geo_address, dict):
            location = geo_address.get("full_address", "") or geo_address.get("address", "")
        else:
            location = ""
        
        # Fallback to other location fields
        if not location:
            location = event.get("location") or event.get("venue") or event.get("address") or ""
        
        row = {
            "Event Name": event.get("name", ""),
            "Luma Event ID": event_id,
            "Start Date & Time": parse_datetime(event.get("start_at")),
            "End Date & Time": parse_datetime(event.get("end_at")),
            "Event Price": price,
            "Is Free?": is_free,
            "Topic Category": event.get("topic_category") or event.get("category") or event.get("topic") or "",
            "Location": location,
            "Total Attendees": total_attendees
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def build_attendees_df(events: List[Dict[str, Any]], guests_by_event: Dict[str, List[Dict[str, Any]]], events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build the attendees DataFrame with required columns.
    
    Args:
        events: List of event dictionaries
        guests_by_event: Dictionary mapping event_id to list of guests
        events_df: Optional pre-computed events DataFrame (preferred for accurate Event Type)
        
    Returns:
        DataFrame with attendee data
    """
    rows = []
    
    # Create event lookup - use id or api_id as key
    event_lookup = {}
    for event in events:
        event_id = event.get("id", "") or event.get("api_id", "")
        if event_id:
            event_lookup[event_id] = event
    
    # Create lookup from events_df if provided (more reliable for Event Type)
    event_type_lookup = {}
    event_name_lookup = {}
    event_date_lookup = {}
    if events_df is not None:
        for _, row in events_df.iterrows():
            event_id = row.get("Luma Event ID", "")
            if event_id:
                event_type_lookup[event_id] = row.get("Is Free?", "Free")
                event_name_lookup[event_id] = row.get("Event Name", "")
                event_date_lookup[event_id] = row.get("Start Date & Time", "")
    
    for event_id, guests in guests_by_event.items():
        event = event_lookup.get(event_id, {})
        
        # Use events_df if available, otherwise calculate
        if event_id in event_type_lookup:
            event_name = event_name_lookup[event_id]
            event_type = event_type_lookup[event_id]
            event_start = event_date_lookup[event_id]
        else:
            event_name = event.get("name", "")
            event_price = parse_price(event, guests)  # Pass guests to check ticket prices
            event_type = "Free" if event_price == 0 else "Paid"
            event_start = parse_datetime(event.get("start_at"))
        
        for guest in guests:
            # Get email - API has both email and user_email
            email = guest.get("email") or guest.get("user_email") or ""
            if not email:
                continue
            
            # Get name - API has name, user_name, or user_first_name + user_last_name
            full_name = (
                guest.get("name") or
                guest.get("user_name") or
                f"{guest.get('user_first_name', '')} {guest.get('user_last_name', '')}".strip() or
                ""
            )
            
            registered_at = parse_datetime(guest.get("registered_at") or guest.get("rsvp_at"))
            checked_in_at = parse_datetime(guest.get("checked_in_at") or guest.get("check_in_at"))
            checked_in = bool(checked_in_at)
            
            # Get ticket type from event_ticket.name
            event_ticket = guest.get("event_ticket", {})
            if isinstance(event_ticket, dict):
                ticket_type = event_ticket.get("name", "")
            else:
                ticket_type = ""
            
            # Fallback to other ticket fields
            if not ticket_type:
                ticket_type = guest.get("ticket_type") or guest.get("ticket_name") or ""
            
            # Mark as flaked if: registered but didn't check in, OR approval_status is declined/cancelled
            approval_status = guest.get("approval_status", "")
            is_declined = approval_status in ["declined", "cancelled"]
            flaked = bool((registered_at and not checked_in_at) or is_declined)
            
            attendee_id = f"{event_id}-{email}"
            
            row = {
                "Attendee ID": attendee_id,
                "Email": email,
                "Full Name": full_name,
                "Luma Event ID": event_id,
                "Event Name": event_name,
                "RSVP Time": registered_at,
                "Check-in Timestamp": checked_in_at,
                "Checked In": checked_in,
                "Ticket Type": ticket_type,
                "Event Type": event_type,
                "Event Date": event_start,
                "Flaked?": flaked
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def build_people_df(attendees_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the people DataFrame aggregated from attendees.
    
    Args:
        attendees_df: DataFrame with attendee data
        events_df: DataFrame with event data
        
    Returns:
        DataFrame with person data
    """
    if attendees_df.empty:
        return pd.DataFrame(columns=[
            "Email", "Full Name", "First Event Date", "Most Recent Event Date",
            "Cohort Month", "Total Events Attended", "Event Types Attended",
            "Days Since Last Event", "Churned?"
        ])
    
    # Normalize email to lowercase for deduplication
    attendees_df = attendees_df.copy()
    attendees_df["email_lower"] = attendees_df["Email"].str.lower()
    
    # Group by email (lowercase)
    people_rows = []
    
    for email_lower, group in attendees_df.groupby("email_lower"):
        # Get first email (preserve original case from first occurrence)
        email = group.iloc[0]["Email"]
        full_name = group.iloc[0]["Full Name"]
        
        # Get all registered_at dates
        registered_dates = group["RSVP Time"].dropna()
        
        if registered_dates.empty:
            first_event_date = None
            most_recent_event_date = None
            cohort_month = ""
            days_since_last = None
        else:
            # Convert to datetime for calculations
            registered_dt = pd.to_datetime(registered_dates)
            first_event_date = registered_dt.min().isoformat()
            most_recent_event_date = registered_dt.max().isoformat()
            
            # Cohort month from first event
            first_dt = pd.to_datetime(first_event_date)
            cohort_month = first_dt.strftime("%Y-%m")
            
            # Days since last event - ensure timezone consistency
            today = pd.Timestamp.now(tz=None)  # Use timezone-naive timestamp
            most_recent_dt = pd.to_datetime(most_recent_event_date)
            
            # Convert timezone-aware datetime to timezone-naive for comparison
            if most_recent_dt.tz is not None:
                most_recent_dt = most_recent_dt.tz_convert('UTC').tz_localize(None)
            
            days_since_last = (today - most_recent_dt).days
        
        # Count events attended - only checked-in events if check-in data exists
        has_checkin_data = group["Check-in Timestamp"].notna().any()
        if has_checkin_data:
            # Only count checked-in events
            checked_in_events = group[group["Checked In"] == True]
            total_events_attended = len(checked_in_events)
        else:
            # Count all registrations if no check-in data
            total_events_attended = len(group)
        
        # Event types attended
        event_types = set(group["Event Type"].dropna().unique())
        event_types_attended = ",".join(sorted(event_types)) if event_types else ""
        
        # Churned? - 3 months (90 days) since last event
        if days_since_last is not None:
            churned = "Yes" if days_since_last > 90 else "No"
        else:
            churned = ""
        
        row = {
            "Email": email,
            "Full Name": full_name,
            "First Event Date": first_event_date,
            "Most Recent Event Date": most_recent_event_date,
            "Cohort Month": cohort_month,
            "Total Events Attended": total_events_attended,
            "Event Types Attended": event_types_attended,
            "Days Since Last Event": days_since_last if days_since_last is not None else "",
            "Churned?": churned
        }
        people_rows.append(row)
    
    return pd.DataFrame(people_rows)


def main():
    """Main execution function."""
    print("Starting Luma data export...")
    print(f"Using API key: {LUMA_API_KEY[:20]}...")
    print()
    
    # Fetch all events
    events = fetch_all_events(LUMA_API_KEY)
    
    if not events:
        print("No events found. Exiting.")
        return
    
    # Fetch guests for each event
    guests_by_event = {}
    total_guests = 0
    
    for i, event in enumerate(events, 1):
        # API uses both id and api_id - use id first, fallback to api_id
        event_id = event.get("id", "") or event.get("api_id", "")
        event_name = event.get("name", "Unknown")
        
        if not event_id:
            print(f"  Skipping event {i}: missing ID")
            continue
        
        print(f"Fetching guests for event {i}/{len(events)}: {event_name} ({event_id})")
        guests = fetch_all_guests_for_event(LUMA_API_KEY, event_id)
        guests_by_event[event_id] = guests
        total_guests += len(guests)
        print(f"  Found {len(guests)} guests")
        time.sleep(REQUEST_DELAY)
    
    print()
    print(f"Total guests fetched: {total_guests}")
    print()
    
    # Build DataFrames
    print("Building DataFrames...")
    events_df = build_events_df(events, guests_by_event)
    attendees_df = build_attendees_df(events, guests_by_event, events_df)  # Pass events_df for accurate Event Type
    people_df = build_people_df(attendees_df, events_df)
    
    # Export to CSV
    print("Exporting to CSV files...")
    events_df.to_csv("events.csv", index=False)
    print("  ✓ events.csv")
    
    people_df.to_csv("people.csv", index=False)
    print("  ✓ people.csv")
    
    attendees_df.to_csv("attendees.csv", index=False)
    print("  ✓ attendees.csv")
    
    # Print summary
    print()
    print("=" * 50)
    print("Export Summary")
    print("=" * 50)
    print(f"Number of events exported: {len(events_df)}")
    print(f"Number of unique people: {len(people_df)}")
    print(f"Number of attendee rows: {len(attendees_df)}")
    print()
    print("Files created:")
    print("  - events.csv")
    print("  - people.csv")
    print("  - attendees.csv")
    print()


if __name__ == "__main__":
    main()
