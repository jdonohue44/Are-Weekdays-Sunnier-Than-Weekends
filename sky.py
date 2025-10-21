import requests
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = "TPE4299DF7RUZWJ2EQRZYVFUP"  # Replace with your real API key
LOCATION = "New York,NY,USA"
OUTPUT_FILE = "cloudcover_daily.csv"

# --- YEARLY RANGES ---
RANGES = [
    ("1990-01-01", "1999-12-31"),
    ("2000-01-01", "2006-12-31"),
    ("2007-01-01", "2016-12-31"),  # new 10-year block
    ("2017-01-01", "2021-12-31"),  # additional 5 years
    ("2022-01-01", "2025-01-01"),  # existing range
]

base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
params = {
    "unitGroup": "metric",
    "include": "days",
    "key": API_KEY,
    "contentType": "json"
}

all_records = []

for start_date, end_date in RANGES:
    print(f"\nFetching cloud cover data for {LOCATION} ({start_date} → {end_date})...")
    url = f"{base_url}/{LOCATION}/{start_date}/{end_date}"

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    for day in data.get("days", []):
        date_str = day.get("datetime")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        cloudcover = day.get("cloudcover", None)
        
        weekday_name = date_obj.strftime("%A")
        label = "Weekend" if weekday_name in ["Saturday", "Sunday"] else "Weekday"
        
        all_records.append({
            "date": date_str,
            "cloudcover": cloudcover,
            "day_of_week": weekday_name,
            "label": label
        })

# --- COMBINE WITH EXISTING FILE IF PRESENT ---
try:
    existing_df = pd.read_csv(OUTPUT_FILE)
    print(f"\nMerging with existing data ({len(existing_df)} rows)...")
except FileNotFoundError:
    existing_df = pd.DataFrame(columns=["date", "cloudcover", "day_of_week", "label"])

new_df = pd.DataFrame(all_records)
combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")

# --- SAVE TO CSV ---
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Saved {len(combined_df)} total days of cloud cover data to {OUTPUT_FILE}")
print(combined_df.head())
print(combined_df.tail())

