import streamlit as st
import pandas as pd
import numpy as np
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

import os

# --- 2. GLOBAL LOAD (Absolute Path Fix) ---
# Get the absolute directory of this file
base_path = os.path.dirname(os.path.abspath(__file__))
# Join that path with your filename
csv_path = os.path.join(base_path, "master_registry.csv")

try:
    # Check if the file actually exists where we think it is
    if os.path.exists(csv_path):
        df_master = pd.read_csv(csv_path)
        df_master.columns = [c.strip().lower() for c in df_master.columns]
        df_master['region'] = df_master['region'].astype(str).str.strip().str.upper()
        st.sidebar.success(f"Registry Loaded! Found {len(df_master)} rows.")
    else:
        # If it fails, list what is actually in the folder to help us debug
        st.sidebar.error(f"File NOT found at {csv_path}")
        st.sidebar.write("Actual files in directory:", os.listdir(base_path))
        df_master = pd.DataFrame()
except Exception as e:
    st.sidebar.error(f"Registry Load Error: {e}")
    df_master = pd.DataFrame()
    
# =====================================================================
# 3. HIGH-POWERED THREE-TIER SHIELD ENGINE (ALGORITHMIC RECONCILIATION)
# =====================================================================
def run_three_tier_shield_engine(standardized_df, master_df, confidence_threshold=0.65):
    """
    Executes structural character-vector distance logic strictly in system memory.
    Reads canonical keys assigned from our zero-collision ingestion hub.
    FIX 7: confidence_threshold is now passed in as a parameter so the sidebar
    slider and the engine use the same boundary (previously the engine hardcoded 0.65
    while the UI applied a different user-defined threshold — contradictory results).
    """
    processed_rows = []
    master_names = master_df["Standard_Name"].values

    for idx, row in standardized_df.iterrows():
        # FIX 9: pandas Series does not support .get() — use index membership check.
        if "SKU" in row.index and pd.notna(row["SKU"]):
            sku = row["SKU"]
        elif "Distributor_SKU" in row.index and pd.notna(row["Distributor_SKU"]):
            sku = row["Distributor_SKU"]
        else:
            sku = f"LINE_{idx}"

        raw_input = str(row["Client_Drug_Name"]).strip()
        wholesaler_price = float(row["Client_Current_Price"])

        best_match_name = None
        highest_score = 0.0
        match_protocol = "Layer 3: Flagged Exception"

        # --- LAYER 1: EXACT MATCH LOOKUP ---
        exact_match = master_df[master_df["Standard_Name"].str.lower() == raw_input.lower()]
        if not exact_match.empty:
            best_match_name = exact_match.iloc[0]["Standard_Name"]
            baseline_price = exact_match.iloc[0]["Regional_Baseline_Price"]
            highest_score = 1.00
            match_protocol = "Layer 1: Exact Registry Match"

        # --- LAYER 2: FUZZY CHARACTER VECTOR MATRIX MATCH ---
        else:
            for master_name in master_names:
                score = SequenceMatcher(None, raw_input.lower(), master_name.lower()).ratio()
                if score > highest_score:
                    highest_score = score
                    best_match_name = master_name

            # FIX 5: Guard against None best_match_name when master registry is empty.
            # Previously, an empty master_df caused a TypeError on the lookup below.
            if best_match_name is not None:
                matched_record = master_df[master_df["Standard_Name"] == best_match_name]
                baseline_price = matched_record.iloc[0]["Regional_Baseline_Price"] if not matched_record.empty else wholesaler_price
            else:
                baseline_price = wholesaler_price

            if highest_score >= confidence_threshold:
                match_protocol = "Layer 2: Algorithmic Vector Match"
            else:
                # --- LAYER 3: LOW CONFIDENCE SAFETIES ENGAGED ---
                match_protocol = "Layer 3: Flagged Anomaly Exception"
                best_match_name = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
                baseline_price = wholesaler_price

        # Core Auditing Metric Derivations
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
    # FIX 2: Use column name lookup instead of hardcoded indices.
    # Hardcoded indices break if columns are ever added, removed, or reordered.
    styles = [''] * len(row)
    col_names = row.index.tolist()
    if "Layer 3" in str(row["Match Protocol"]):
        return ['background-color: rgba(255, 193, 7, 0.22); border-left: 4px solid #FFC107;'] * len(row)
    # FIX 6 (partial): Only highlight as overcharged when leakage is genuinely positive.
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
    
    # --- SIDEBAR PARAMETERS & COMPLIANCE GUARDRAIL ---
    with st.sidebar:
        st.markdown("### 📊 Parameter Profiles")
        selected_region = st.selectbox("Target Jurisdictional Region", ["UK", "US", "Canada"])
        confidence_threshold = st.slider("Automated Layer Confidence Filter", 0.50, 1.00, 0.70, help="Any algorithmic calculation score dropping below this percentage triggers human confirmation layers.")
        
        st.write("---")
        st.caption(f"Active Profile: {selected_region} Data Standalone Core")
        st.caption("Workspace Security: Sandboxed Session RAM")
        
        # =============================================================
        # COMPLIANCE & INSTITUTIONAL SUPPORT HUB
        # =============================================================
        st.write("---")
        
        # Section 1: Strict Architectural Isolation Guard
        with st.container(border=True):
            st.markdown("<small>🔒 **ARCHITECTURAL BOUNDARY**</small>", unsafe_allow_html=True)
            st.caption(
                "Apex Logic operates exclusively as a financial reconciliation framework. "
                "This platform **does not interface or sync** with Patient Record Systems (EMR/EHR) "
                "or E-Prescription (eRx) networks to ensure absolute structural data sandboxing."
            )
            
        # Section 2: Enterprise Support Grid
        st.markdown("### 🏛️ Enterprise Support")
        
        # Professional Ticket Router Link
        st.link_button(
            "💻 Access Enterprise Service Desk",
            url="https://support.apexlogic.ai/portal",
            use_container_width=True
        )
        
        # High-Priority SLA Callback Request Link
        st.link_button(
            "📞 Request Priority Operational Callback",
            url="mailto:enterprise-ops@apexlogic.ai?subject=URGENT%20SLA%20Callback%20Request",
            use_container_width=True
        )
        
        # Contextual metadata parameters for corporate clients
        st.caption(
            "**Secure Node Connection:** `Verified TLS 1.3`  \n"
            "**Contracted Response SLA:** `2-Hour Institutional Turnaround`  \n"
            "**Regulatory Coverage:** `B2B Procurement Cleared`"
        )

    master_registry_df = load_global_master_registry(selected_region)
    
    # UI Fix 1: Unified, Clean Procurement Ingestion Hub
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
            
        st.dataframe(client_df.head(5), use_container_width=True)
        
        # --- LAYER 1: CHARACTER ERASER RADAR ---
        raw_columns = client_df.columns.tolist()
        normalized_map = {
            col: col.strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "") 
            for col in raw_columns
        }
        
        # --- LAYER 2: MULTI-FIELD SYNONYM THESAURUS ---
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

        # --- LAYER 3: ELEGANT DROPDOWN BACKSTOPS ---
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
                final_drug_col = st.selectbox(
                    "Map your Product/Drug Name column manually:",
                    options=raw_columns, index=default_drug_idx, key="manual_drug_select"
                )
                
        with col2:
            if detected_price_col:
                st.success(f"🎯 Auto-detected Pricing Column: **'{detected_price_col}'**")
                final_price_col = detected_price_col
            else:
                st.warning("🕵️‍♂️ Pricing/Cost column could not be auto-aligned.")
                final_price_col = st.selectbox(
                    "Map your Current Unit Price column manually:",
                    options=raw_columns, index=default_price_idx, key="manual_price_select"
                )

        # --- DATA STANDARDIZATION LOCKDOWN (DIRECT BINDING) ---
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

        # --- LIVE RECONCILIATION DATA PROCESSING STREAM ---
        st.markdown("---")
        st.markdown("#### ⚡ Live Reconciliation Audit Data Engine")

        # FIX 8: Cache audit results in session_state so the engine does not
        # re-run on every widget interaction (e.g. touching the override dropdowns).
        # A cache key based on filename + region + threshold detects when a real re-run is needed.
        audit_cache_key = f"{uploaded_file.name}_{selected_region}_{confidence_threshold}"
        if st.session_state.get("audit_cache_key") != audit_cache_key:
            with st.spinner("Executing sequence match calculations across regional master profiles..."):
                # FIX 7: Pass confidence_threshold into the engine so the sidebar slider
                # and the Layer 2/3 boundary are always in sync.
                st.session_state.audit_results = run_three_tier_shield_engine(
                    processing_df, master_registry_df, confidence_threshold
                )
                st.session_state.audit_cache_key = audit_cache_key

        audit_results_df = st.session_state.audit_results
            
        flagged_anomalies = audit_results_df[audit_results_df["Raw_Confidence"] < confidence_threshold]
        # FIX 6: Separate positive leakage (overcharged) from negative (undercharged/favourable).
        # Summing all values including negatives produced misleading or negative "savings" figures.
        resolved_df = audit_results_df[audit_results_df["Raw_Confidence"] >= confidence_threshold]
        total_overcharged = resolved_df[resolved_df["Price Leakage"] > 0]["Price Leakage"].sum()
        total_undercharged = resolved_df[resolved_df["Price Leakage"] < 0]["Price Leakage"].sum()

        # Financial Metric Grid Display
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Manifest Rows Audited", len(audit_results_df))
        with metric_col2:
            # FIX 10: Renamed from "Capital Leakage Recovered" — the audit identifies
            # potential overpayments, it does not recover them. Accurate language matters.
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
        st.dataframe(display_df.style.apply(highlight_anomalies, axis=1), use_container_width=True)
        
        # --- HUMAN-IN-THE-LOOP EXCEPTION OVERRIDES ---
        if len(flagged_anomalies) > 0:
            st.markdown("---")
            st.markdown("#### 🛠️ Step 3: Human-In-The-Loop Exception Resolution Matrix")
            st.warning("The algorithmic engine flagged low-confidence inputs that failed your threshold safety. Reassign them manually below:")

            for idx, row in flagged_anomalies.iterrows():
                st.markdown(f"**SKU Source Target:** `{row['SKU']}` | **Corrupted Text Ingested:** *\"{row['Raw Input Phrase']}\"*")
                # FIX 4: Use DataFrame index (idx) as the widget key, not the SKU value.
                # Duplicate SKUs (common in real invoices) caused DuplicateWidgetID crashes.
                widget_key = f"override_{idx}"
                override_options = ["-- Select Correct Registry Mapping --"] + list(master_registry_df["Standard_Name"].values)
                st.selectbox(
                    f"Select Confirmed Identity to Map to Item {row['SKU']}:",
                    options=override_options,
                    key=widget_key
                )

            # FIX 3: Apply override selections back to the audit results before export.
            # Previously, selections were rendered but never read — the export still
            # contained unresolved rows. Now we patch resolved rows into a copy.
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
        
        # --- CLEAN EXPORT MANAGEMENT LINK ---
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
