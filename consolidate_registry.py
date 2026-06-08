import pandas as pd
import json
import xml.etree.ElementTree as ET
import os

def consolidate_data():
    master_list = []
    
    # 1. PROCESS US (FDA JSON)
    print("Processing FDA Data...")
    with open('data/raw/fda_ndc_raw.json', 'r') as f:
        data = json.load(f)
        for entry in data.get('results', []):
            for pack in entry.get('packaging', []):
                master_list.append({
                    'Country': 'US',
                    'Drug_Name': entry.get('brand_name'),
                    'Generic_Name': entry.get('generic_name'),
                    'Identifier': pack.get('package_ndc'),
                    'Dosage': entry.get('dosage_form')
                })

    # 2. PROCESS CANADA (Health Canada CSVs)
    # We focus on the Product and Packaging files
    print("Processing Canada Data...")
    df_prod = pd.read_csv('data/raw/drug.txt', sep='^') # Adjust separator if needed
    df_pack = pd.read_csv('data/raw/package.txt', sep='^')
    df_ca = pd.merge(df_prod, df_pack, on='drug_code')
    
    for _, row in df_ca.iterrows():
        master_list.append({
            'Country': 'CA',
            'Drug_Name': row['brand_name'],
            'Generic_Name': 'N/A', # Map accordingly
            'Identifier': row['drug_identification_number'],
            'Dosage': row['dosage_form_name']
        })

    # 3. PROCESS UK (NHS XML)
    # This will need slight adjustment based on your specific XML tags
    print("Processing UK Data...")
    tree = ET.parse('data/raw/nhs_dmd.xml')
    root = tree.getroot()
    # Logic to iterate through NHS XML elements and append to master_list...

    # Create Final Master Registry
    final_df = pd.DataFrame(master_list)
    final_df.to_csv('data/processed/master_registry.csv', index=False)
    print("Success! master_registry.csv created.")

if __name__ == "__main__":
    consolidate_data()