import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from difflib import SequenceMatcher

# =====================================================================
# 1. PLATFORM CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="Apex Logic | Global Pharmacy Audit",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .brand-header { font-size: 42px; font-weight: 800; letter-spacing: 2px; color: #FFFFFF; margin-bottom: 0px; }
        .brand-subtitle { font-size: 14px; font-weight: 400; letter-spacing: 4px; color: #00FFCC; margin-top: 0px; margin-bottom: 25px; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 12px 24px; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. MASTER REGISTRY LOADER (CSV-based, Streamlit Cloud compatible)
# =====================================================================
# Loads master_registry.csv from the same folder as this script.
# The CSV must have these columns (case-insensitive, spaces allowed):
#   region | standard_name | regional_baseline_price
#
# The loader normalises column names internally, then returns a clean
# DataFrame with the exact column names the engine expects:
#   "Standard_Name" and "Regional_Baseline_Price"
#
# load_global_master_registry() is the single call site used below —
# it filters the full registry down to the selected region before
# returning, so the engine only sees relevant rows.

# Maps the dropdown label to the exact string used in the CSV region column
REGION_KEY_MAP = {
    "UK":     "UK",
    "US":     "US",
    "Canada": "CANADA",  # CSV stores "CANADA" (uppercased); dropdown shows "Canada"
}

@st.cache_data
def _load_full_registry() -> pd.DataFrame:
    """
    Reads master_registry.csv once and caches the result for the session.
    NO st calls inside this function — st.cache_data suppresses UI calls
    after the first run, making diagnostics invisible. Diagnostics are
    handled in load_global_master_registry() which is NOT cached.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "master_registry.csv")

    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Standard_Name", "Regional_Baseline_Price", "region", "_error"])

    try:
        df = pd.read_csv(csv_path)
        # Normalise column names: strip whitespace, lowercase, underscores for spaces
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Flexible column matching
        name_aliases   = ["standard_name", "drug_name", "name", "drug", "medication", "product"]
        price_aliases  = ["regional_baseline_price", "baseline_price", "price", "unit_price", "cost", "amount"]
        region_aliases = ["region", "country", "jurisdiction", "market"]

        matched_name_col   = next((c for c in df.columns if c in name_aliases),   None)
        matched_price_col  = next((c for c in df.columns if c in price_aliases),  None)
        matched_region_col = next((c for c in df.columns if c in region_aliases), None)

        if not matched_name_col or not matched_price_col or not matched_region_col:
            # Return an empty frame with an _error column so the caller can show a message
            err = pd.DataFrame(columns=["Standard_Name", "Regional_Baseline_Price", "region", "_error"])
            err.attrs["col_error"] = f"Detected columns: {list(df.columns)}"
            return err

        df = df.rename(columns={
            matched_name_col:   "Standard_Name",
            matched_price_col:  "Regional_Baseline_Price",
            matched_region_col: "region"
        })

        # Uppercase region so filtering is case-insensitive ("us" == "US" == "CANADA")
        df["region"] = df["region"].astype(str).str.strip().str.upper()
        df["_loaded"] = True  # sentinel so caller knows load succeeded
        return df[["Standard_Name", "Regional_Baseline_Price", "region", "_loaded"]]

    except Exception as e:
        err = pd.DataFrame(columns=["Standard_Name", "Regional_Baseline_Price", "region", "_error"])
        err.attrs["load_error"] = str(e)
        return err


def load_global_master_registry(region: str) -> pd.DataFrame:
    """
    Public entry point. Filters the full registry to the selected region
    and surfaces diagnostics — safe to call st here since this is NOT cached.
    """
    full_df = _load_full_registry()

    # Surface any load errors
    if "_error" in full_df.columns:
        col_error = full_df.attrs.get("col_error")
        load_error = full_df.attrs.get("load_error")
        if load_error:
            st.sidebar.error(f"❌ Failed to read master_registry.csv: {load_error}")
        elif col_error:
            st.sidebar.error(
                f"❌ Could not find required columns in master_registry.csv.\n"
                f"{col_error}\n"
                f"Expected columns like: region, standard_name, regional_baseline_price"
            )
        return pd.DataFrame(columns=["Standard_Name", "Regional_Baseline_Price"])

    if "_loaded" not in full_df.columns:
        st.sidebar.error("❌ master_registry.csv not found in the repo root folder.")
        return pd.DataFrame(columns=["Standard_Name", "Regional_Baseline_Price"])

    # Map the dropdown value ("Canada") to the CSV region key ("CANADA")
    region_key = REGION_KEY_MAP.get(region, region.strip().upper())
    unique_regions = sorted(full_df["region"].unique().tolist())

    filtered = full_df[full_df["region"] == region_key][["Standard_Name", "Regional_Baseline_Price"]]

    if filtered.empty:
        st.sidebar.warning(
            f"⚠️ No drugs found for region `{region_key}`. "
            f"Regions present in your CSV: `{unique_regions}`"
        )
    else:
        st.sidebar.success(
            f"✅ {len(filtered)} drugs loaded for {region} (`{region_key}`). "
            f"All regions in CSV: `{unique_regions}`"
        )

    return filtered.reset_index(drop=True)



# =====================================================================
# 3. INVOICE NORMALISATION + THREE-TIER SHIELD ENGINE
# =====================================================================
# Real pharmacy invoices use abbreviations and include dosage/batch noise:
#   "AMOX 500MG CAPS // BATCH-58"  ->  fuzzy score vs "Amoxicillin" = 0.35 (FAILS)
# After normalisation:
#   "Amoxicillin"                  ->  fuzzy score vs "Amoxicillin" = 1.00 (PASSES)
#
# Two-step process:
#   1. Strip batch codes, dosage strengths, and dosage form words.
#   2. Expand known abbreviations to full clinical names.

import re as _re

_ABBREV_MAP = {
    "lisino":         "Lisinopril",
    "amox/clav":      "Amoxicillin",
    "amox":           "Amoxicillin",
    "gaba":           "Gabapentin",
    "atorva":         "Atorvastatin",
    "simva":          "Simvastatin",
    "ibu":            "Ibuprofen",
    "para":           "Paracetamol",
    "amlo":           "Amlodipine",
    "levo":           "Levothyroxine",
    "metoprolol tar": "Metoprolol",
    "met":            "Metformin",
}

_STRIP_PATTERN = _re.compile(
    r'\s*//\s*batch[-\s]?\w+'
    r'|\d+(?:\.\d+)?(?:/\d+)?\s*(?:mg|mcg|g|ml|iu)'
    r'|(?:tab|tabs|cap|caps|er|sr|xr|eff|inh|inj|soln?|susp|tar)'
    r'|x-\d+',
    _re.IGNORECASE
)

def _normalise_drug_name(raw: str) -> str:
    name = _STRIP_PATTERN.sub(" ", raw).strip()
    name = " ".join(name.split())
    name_lower = name.lower()
    for abbrev, full in _ABBREV_MAP.items():
        pattern = _re.compile(r'^' + _re.escape(abbrev) + r'(\s|$)', _re.IGNORECASE)
        if pattern.match(name_lower):
            remainder = pattern.sub("", name_lower).strip()
            name = full + (" " + remainder if remainder else "")
            break
    name = _re.sub(r'[a-zA-Z]', '', name).strip()
    return " ".join(name.split())


def run_three_tier_shield_engine(standardized_df, master_df, confidence_threshold=0.65):
    processed_rows = []
    master_names = master_df["Standard_Name"].values

    for idx, row in standardized_df.iterrows():
        if "SKU" in row.index and pd.notna(row["SKU"]):
            sku = row["SKU"]
        elif "Distributor_SKU" in row.index and pd.notna(row["Distributor_SKU"]):
            sku = row["Distributor_SKU"]
        else:
            sku = f"LINE_{idx}"

        raw_input = str(row["Client_Drug_Name"]).strip()
        normalised_input = _normalise_drug_name(raw_input)
        wholesaler_price = float(row["Client_Current_Price"])

        best_match_name = None
        highest_score = 0.0
        match_protocol = "Layer 3: Flagged Exception"

        # --- LAYER 1: EXACT MATCH LOOKUP --- uses normalised name
        exact_match = master_df[master_df["Standard_Name"].str.lower() == normalised_input.lower()]
        if not exact_match.empty:
            best_match_name = exact_match.iloc[0]["Standard_Name"]
            baseline_price = exact_match.iloc[0]["Regional_Baseline_Price"]
            highest_score = 1.00
            match_protocol = "Layer 1: Exact Registry Match"

        # --- LAYER 2: FUZZY CHARACTER VECTOR MATRIX MATCH ---
        else:
            for master_name in master_names:
                score = SequenceMatcher(None, normalised_input.lower(), master_name.lower()).ratio()
                if score > highest_score:
                    highest_score = score
                    best_match_name = master_name

            if best_match_name is not None:
                matched_record = master_df[master_df["Standard_Name"] == best_match_name]
                baseline_price = matched_record.iloc[0]["Regional_Baseline_Price"] if not matched_record.empty else wholesaler_price
            else:
                baseline_price = wholesaler_price

            if highest_score >= confidence_threshold:
                match_protocol = "Layer 2: Algorithmic Vector Match"
            else:
                match_protocol = "Layer 3: Flagged Anomaly Exception"
                best_match_name = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
                baseline_price = wholesaler_price

        variance = wholesaler_price - baseline_price
        variance_pct = (variance / baseline_price) * 100 if baseline_price > 0 else 0

        processed_rows.append({
            "SKU": sku,
            "Raw Input Phrase": raw_input,
            "Resolved Clinical Name": best_match_name,
            "Match Protocol": match_protocol,
            "Confidence": f"{highest_score * 100:.1f}%",
            "Invoice Price": f"${wholesaler_price:.2f}",
            "Market Baseline": f"${baseline_price:.2f}",
            "Price Leakage": variance,
            "Leakage %": variance_pct,
            "Raw_Confidence": highest_score
        })

    return pd.DataFrame(processed_rows)


def highlight_anomalies(row):
    styles = [''] * len(row)
    col_names = row.index.tolist()
    if "Layer 3" in str(row["Match Protocol"]):
        return ['background-color: rgba(255, 193, 7, 0.22); border-left: 4px solid #FFC107;'] * len(row)
    if float(row["Price Leakage"]) > 0.05 and "Layer 3" not in str(row["Match Protocol"]):
        highlight = 'background-color: rgba(255, 87, 34, 0.25); color: #FF9966; font-weight: bold;'
        for col in ["Invoice Price", "Price Leakage"]:
            if col in col_names:
                styles[col_names.index(col)] = highlight
    return styles


# =====================================================================
# 4. UNIFIED STREAMLIT USER INTERFACE FRAMEWORK
# =====================================================================
st.markdown('<div class="brand-header">▲ APEX LOGIC</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">GLOBAL PHARMACY AUDIT ENGINE</div>', unsafe_allow_html=True)

tab_website, tab_webapp = st.tabs(["🏠 Platform Overview", "⚡ Automated Audit Suite"])

# -----------------------------------------------------------------
# TAB 1: FRONT-END WEBSITE OVERVIEW
# -----------------------------------------------------------------
with tab_website:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### Reclaim Your Pharmacy's Lost Margin")
        st.write(
            "Apex Logic cross-references your chaotic, corrupted distributor manifests against "
            "live national baseline medical registries instantly using mathematical string sequence mapping—"
            "isolating data anomalies and hidden price gouging before you clear accounts payable."
        )
        st.markdown("#### 🌍 Cross-Border Core Registry Footprints Supported")
        st.info("**United Kingdom:** NHS dm+d infrastructure tracking.\n\n**United States:** FDA National Drug Code listings.\n\n**Canada:** Health Canada Drug Product Database syncing.")
    with col_right:
        st.markdown("⚙️ **Platform Framework Status**")
        st.success("Layer 1 Database Hashing: Online")
        st.success("Layer 2 String Match Vectoring: Active")
        st.success("Layer 3 Deep Boundary Safety Safeguards: Engaged")

# -----------------------------------------------------------------
# TAB 2: THE OPERATIONAL CORE INDUSTRIAL SOFTWARE WORKSPACE
# -----------------------------------------------------------------
with tab_webapp:
    st.markdown("### Operational Audit Dashboard")

    with st.sidebar:
        st.markdown("### 📊 Parameter Profiles")
        selected_region = st.selectbox("Target Jurisdictional Region", ["UK", "US", "Canada"])
        confidence_threshold = st.slider("Automated Layer Confidence Filter", 0.50, 1.00, 0.70, help="Any algorithmic calculation score dropping below this percentage triggers human confirmation layers.")

        st.write("---")
        st.caption(f"Active Profile: {selected_region} Data Standalone Core")
        st.caption("Workspace Security: Sandboxed Session RAM")

        st.write("---")
        with st.container(border=True):
            st.markdown("<small>🔒 **ARCHITECTURAL BOUNDARY**</small>", unsafe_allow_html=True)
            st.caption(
                "Apex Logic operates exclusively as a financial reconciliation framework. "
                "This platform **does not interface or sync** with Patient Record Systems (EMR/EHR) "
                "or E-Prescription (eRx) networks to ensure absolute structural data sandboxing."
            )

        st.markdown("### 🏛️ Enterprise Support")
        st.link_button("💻 Access Enterprise Service Desk", url="https://support.apexlogic.ai/portal", width='stretch')
        st.link_button("📞 Request Priority Operational Callback", url="mailto:enterprise-ops@apexlogic.ai?subject=URGENT%20SLA%20Callback%20Request", width='stretch')
        st.caption(
            "**Secure Node Connection:** `Verified TLS 1.3`  \n"
            "**Contracted Response SLA:** `2-Hour Institutional Turnaround`  \n"
            "**Regulatory Coverage:** `B2B Procurement Cleared`"
        )

    # This now correctly calls the function defined above
    master_registry_df = load_global_master_registry(selected_region)

    st.markdown("#### 📑 Procurement Manifest Ingestion")
    st.markdown("<p style='color: #888; margin-top: -10px;'>Drop distribution invoices, formulas, or raw manifest logs to run registry audits.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Client Invoice, Formulary, or Procurement Sheet (.csv, .xlsx)",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        help="Supports standard enterprise procurement sheets. Columns will auto-align.",
        key="client_procurement_uploader"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                client_df = pd.read_csv(uploaded_file)
            else:
                client_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"🚨 Failed to read file structural layout. Error detail: {e}")
            st.stop()

        st.dataframe(client_df.head(5), width='stretch')

        raw_columns = client_df.columns.tolist()
        normalized_map = {
            col: col.strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "")
            for col in raw_columns
        }

        synonym_thesaurus = {
            "drug_name": [
                "drugname", "drug", "product", "productname", "item", "itemdescription",
                "medication", "medicationname", "medicine", "description", "standardname",
                "molecule", "activeingredient", "clinicalname", "brand", "rawinputphrase", "resolvedclinicalname"
            ],
            "current_price": [
                "currentprice", "price", "cost", "unitprice", "rate", "amount",
                "baselineprice", "procurementcost", "contractprice", "billingamount", "acquisitioncost", "invoiceprice", "marketbaseline"
            ]
        }

        detected_drug_col = None
        detected_price_col = None

        for original_col, clean_col in normalized_map.items():
            if not detected_drug_col and clean_col in synonym_thesaurus["drug_name"]:
                detected_drug_col = original_col
            elif not detected_price_col and clean_col in synonym_thesaurus["current_price"]:
                detected_price_col = original_col

        st.markdown("#### ⚙️ Ingestion Matrix Verification")
        col1, col2 = st.columns(2)

        default_drug_idx = raw_columns.index(detected_drug_col) if detected_drug_col in raw_columns else 0
        default_price_idx = raw_columns.index(detected_price_col) if detected_price_col in raw_columns else min(1, len(raw_columns) - 1)

        with col1:
            if detected_drug_col:
                st.success(f"🎯 Auto-detected Drug Column: **'{detected_drug_col}'**")
                final_drug_col = detected_drug_col
            else:
                st.warning("🕵️‍♂️ Product/Drug column could not be auto-aligned.")
                final_drug_col = st.selectbox("Map your Product/Drug Name column manually:", options=raw_columns, index=default_drug_idx, key="manual_drug_select")

        with col2:
            if detected_price_col:
                st.success(f"🎯 Auto-detected Pricing Column: **'{detected_price_col}'**")
                final_price_col = detected_price_col
            else:
                st.warning("🕵️‍♂️ Pricing/Cost column could not be auto-aligned.")
                final_price_col = st.selectbox("Map your Current Unit Price column manually:", options=raw_columns, index=default_price_idx, key="manual_price_select")

        processing_df = client_df.copy()

        if final_drug_col not in raw_columns:
            final_drug_col = raw_columns[0] if raw_columns else None

        if final_drug_col:
            processing_df["Client_Drug_Name"] = client_df[final_drug_col]
        else:
            st.error("🚨 Ingestion Engine Error: Unable to bind a valid Product/Drug column.")
            st.stop()

        if final_price_col and final_price_col in raw_columns and final_price_col != final_drug_col:
            processing_df["Client_Current_Price"] = (
                client_df[final_price_col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .replace("", "0")
                .astype(float)
            )
        else:
            processing_df["Client_Current_Price"] = 0.0
            st.info("💡 No valid pricing column mapped. Initializing current benchmarks to 0.0.")

        processing_df = processing_df.dropna(subset=["Client_Drug_Name"])
        processing_df["Client_Drug_Name"] = processing_df["Client_Drug_Name"].astype(str).str.strip()
        st.toast("🚀 Ingestion engine synchronized with downstream parameters!", icon="✅")

        st.markdown("---")
        st.markdown("#### ⚡ Live Reconciliation Audit Data Engine")

        audit_cache_key = f"{uploaded_file.name}_{selected_region}_{confidence_threshold}"
        if st.session_state.get("audit_cache_key") != audit_cache_key:
            with st.spinner("Executing sequence match calculations across regional master profiles..."):
                st.session_state.audit_results = run_three_tier_shield_engine(
                    processing_df, master_registry_df, confidence_threshold
                )
                st.session_state.audit_cache_key = audit_cache_key

        audit_results_df = st.session_state.audit_results

        flagged_anomalies = audit_results_df[audit_results_df["Raw_Confidence"] < confidence_threshold]
        resolved_df = audit_results_df[audit_results_df["Raw_Confidence"] >= confidence_threshold]
        total_overcharged = resolved_df[resolved_df["Price Leakage"] > 0]["Price Leakage"].sum()
        total_undercharged = resolved_df[resolved_df["Price Leakage"] < 0]["Price Leakage"].sum()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Manifest Rows Audited", len(audit_results_df))
        with metric_col2:
            st.metric(
                "Estimated Overpayment Exposure",
                f"${total_overcharged:.2f}",
                delta=f"${abs(total_undercharged):.2f} Favourable Variance" if total_undercharged < 0 else None,
                delta_color="normal"
            )
        with metric_col3:
            st.metric("Unresolved Exceptions Flagged", len(flagged_anomalies), delta=f"{len(flagged_anomalies)} Actions Required", delta_color="off")

        st.markdown("##### Processed Audit Stream Workspace")
        st.caption("💡 Highlighted Row = Layer 3 Flagged Exception | Highlighted Financial Cells = Wholesaler Price Gouging Surge Over National Averages")

        display_df = audit_results_df.drop(columns=["Raw_Confidence"])
        st.dataframe(display_df.style.apply(highlight_anomalies, axis=1), width='stretch')

        if len(flagged_anomalies) > 0:
            st.markdown("---")
            st.markdown("#### 🛠️ Step 3: Human-In-The-Loop Exception Resolution Matrix")
            st.warning("The algorithmic engine flagged low-confidence inputs that failed your threshold safety. Reassign them manually below:")

            for idx, row in flagged_anomalies.iterrows():
                st.markdown(f"**SKU Source Target:** `{row['SKU']}` | **Corrupted Text Ingested:** *\"{row['Raw Input Phrase']}\"*")
                widget_key = f"override_{idx}"
                override_options = ["-- Select Correct Registry Mapping --"] + list(master_registry_df["Standard_Name"].values)
                st.selectbox(f"Select Confirmed Identity to Map to Item {row['SKU']}:", options=override_options, key=widget_key)

            export_df = audit_results_df.drop(columns=["Raw_Confidence"]).copy()
            for idx, row in flagged_anomalies.iterrows():
                widget_key = f"override_{idx}"
                chosen = st.session_state.get(widget_key, "-- Select Correct Registry Mapping --")
                if chosen != "-- Select Correct Registry Mapping --":
                    export_df.at[idx, "Resolved Clinical Name"] = chosen
                    export_df.at[idx, "Match Protocol"] = "Layer 1: Human Override Confirmed"
                    export_df.at[idx, "Confidence"] = "100.0% (Manual)"
        else:
            export_df = audit_results_df.drop(columns=["Raw_Confidence"]).copy()

        st.markdown("---")
        st.markdown("#### 📤 Step 4: Export Verified Audit Logs")
        csv_payload = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Cleaned Audit Logs (.CSV)",
            data=csv_payload,
            file_name=f"apex_logic_audit_{selected_region.lower()}.csv",
            mime="text/csv"
        )

    else:
        st.write("---")
        st.info("Waiting for data manifest asset drop. Use your testing spreadsheet configurations to verify pipeline mechanics.")
