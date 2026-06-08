import requests
import pandas as pd

# The raw URLs for the government databases
REGISTRY_URLS = {
    "US": "https://raw.githubusercontent.com/username/repo/main/fda_master.csv",
    "UK": "https://raw.githubusercontent.com/username/repo/main/nhs_master.csv",
    "CA": "https://raw.githubusercontent.com/username/repo/main/health_canada_master.csv"
}

def check_for_updates(country):
    # Get the Last-Modified header from the GitHub raw URL
    response = requests.head(REGISTRY_URLS[country])
    last_modified = response.headers.get('Last-Modified')
    
    # If last_modified > our local cache timestamp, trigger download
    if is_newer_than_cache(last_modified):
        download_and_cache(country)
