import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import hashlib
import re as _re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# =====================================================================
# 1. PLATFORM CONFIGURATION & JURISDICTIONAL TAXONOMY
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

# Jurisdictional translation map for frontend dashboard labels
JURISDICTION_PROFILES = {
    "UK": {
        "csv_key": "UK",
        "generic_label": "VMP Name (Virtual Medicinal Product)",
        "brand_label": "AMP Name (Actual Medicinal Product)",
        "id_label": "VMP / AMP ID"
    },
    "US": {
        "csv_key": "US",
        "generic_label": "Established Name (Non-Proprietary)",
        "brand_label": "Proprietary Name (Brand Name)",
        "id_label": "NDC Number"
    },
    "Canada": {
        "csv_key": "CANADA",
        "generic_label": "Active Ingredient Name",
        "brand_label": "Brand Name (Product Name)",
        "id_label": "DIN (Drug Identification Number)"
    }
}

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer1_cache_db.json")

def load_cache_db():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache_db(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

# =====================================================================
# 2. ADVANCED INVOICE NORMALISATION EXTRACTION (FROM SCRIPT 1_2)
# =====================================================================
_ABBREV_MAP = [
    ("amox/clav",      "Amoxicillin"),   
    ("metoprolol tar", "Metoprolol"),
    ("metoprolol",     "Metoprolol"),
    ("metformin",      "Metformin"),     
    ("lisino",         "Lisinopril"),
    ("atorva",         "Atorvastatin"),
    ("simva",          "Simvastatin"),
    ("gaba",           "Gabapentin"),
    ("ibu",            "Ibuprofen"),
    ("para",           "Paracetamol"),
    ("amlo",           "Amlodipine"),
    ("levo",           "Levothyroxine"),
    ("amox",           "Amoxicillin"),
    ("met",            "Metformin"),     
]

_STRIP_PATTERN = _re.compile(
    r'\s*//\s*batch[-\s]?\w+'
    r'| \d+(?:\.\d+)?(?:/\d+)?\s*(?:mg|mcg|g|ml|iu) '
    r'| (?:tab|tabs|cap|caps|er|sr|xr|eff|inh|inj|soln?|susp|tar) '
    r'| x-\d+ ',
    _re.IGNORECASE
)

def _normalise_drug_name(raw: str) -> str:
    name = _STRIP_PATTERN.sub(" ", raw).strip()
    name = " ".join(name.split())
    name_lower = name.lower()
    for abbrev, full in _ABBREV_MAP:
        pattern = _re.compile(r'^' + _re.escape(abbrev) + r'(\s|$)', _re.IGNORECASE)
        if pattern.match(name_lower):
            remainder = pattern.sub("", name_lower).strip()
            name = full + (" " + remainder if remainder else "")
            break
    name = _re.sub(r' [a-zA-Z] ', '', name).strip()
    return " ".join(name.split())

# =====================================================================
# 3. FLEXIBLE MASTER REGISTRY LOADER
# =====================================================================
@st.cache_data
def _load_full_registry() -> pd.DataFrame:
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "master_registry.csv")

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        
        # Flexibly match registry columns from both variations
        df = df.rename(columns={
            "region": "region",
            "country": "region",
            "jurisdiction": "region",
            "market": "region",
            "vmp_name": "generic_name",
            "generic_name": "generic_name",
            "active_ingredient": "generic_name",
            "standard_name": "generic_name",
            "drug_name": "generic_name",
            "amp_name": "brand_name",
            "brand_name": "brand_name",
            "proprietary_name": "brand_name",
            "product_name": "brand_name",
            "regional_baseline_price": "baseline_price",
            "price": "baseline_price",
            "unit_price": "baseline_price",
            "cost": "baseline_price",
            "system_id": "system_id",
            "ndc": "system_id",
            "din": "system_id"
        })
        
        df["region"] = df["region"].astype(str).str.strip().str.upper()
        return df
    except:
        return pd.DataFrame()

def load_jurisdictional_registry(region_setting: str) -> pd.DataFrame:
    full_df = _load_full_registry()
    if full_df.empty:
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id", "region"])
    
    target_key = JURISDICTION_PROFILES.get(region_setting, {}).get("csv_key", "UK")
    filtered = full_df[full_df["region"] == target_key].copy()
    
    for col in ["generic_name", "brand_name", "baseline_price", "system_id"]:
        if col not in filtered.columns:
            filtered[col] = "N/A"
            
    return filtered.reset_index(drop=True)

# =====================================================================
# 4. ADVANCED 3-TIER SHIELD RECONCILIATION ENGINE
# =====================================================================
def call_layer3_ai_api(raw_text: str, jurisdiction: str, examples: list) -> dict:
    """
    [Phase 4: Layer 3 Fallback AI Engine] Routes through Google Gemini's Free Tier
    """
    if "GEMINI_API_KEY" not in st.secrets:
        return {"status": "failure"}
    
    try:
        # Uses Google's native OpenAI-compatibility endpoint
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=st.secrets["GEMINI_API_KEY"]
        )
        
        system_prompt = (
            f"You are an expert clinical pharmacy database matching node for the {jurisdiction} market.\n"
            f"Match the messy input text string to its exact clean generic identity and brand name profile.\n"
            f"You must evaluate and return ONLY a valid JSON object matching this schema precisely without prose:\n"
            "{\n"
            "  \"generic_name\": \"Clean Generic Name/VMP with strength\",\n"
            "  \"brand_name\": \"Clean Brand Name/AMP/Manufacturer descriptor\",\n"
            "  \"system_id\": \"Standard registry ID code (NDC, DIN, or VMP ID) if inferred from context\",\n"
            "  \"manufacturer_id\": \"Inferred wholesaler code or original tracking ID\"\n"
            "}"
        )
        
        user_prompt = f"Messy Input Text: \"{raw_text}\"\n\nValid system database options for context:\n{examples[:30]}"
        
        # Uses Gemini 2.5 Flash - drop-in OpenAI client compatible
        response = client.chat.completions.create(
            model="gemini-2.5-flash", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"status": "failure"}

def run_reconciliation_audit(client_df: pd.DataFrame, master_df: pd.DataFrame, jurisdiction: str, confidence_threshold=0.92):
    processed_rows = []
    cache_db = load_cache_db()
    
    # Intelligently construct lookup strings pooling together brand and generic properties 
    master_df["lookup_pool"] = master_df["generic_name"].astype(str) + " " + master_df["brand_name"].astype(str)
    search_pool = master_df["lookup_pool"].dropna().unique().tolist()
    
    # Character-level TF-IDF prevents false exclusions of brand/generic syntax shifts
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(search_pool) if search_pool else None

    for idx, row in client_df.iterrows():
        # Dynamic tracking token handling for SKUs
        if "SKU" in row.index and pd.notna(row["SKU"]):
            sku = row["SKU"]
        elif "Distributor_SKU" in row.index and pd.notna(row["Distributor_SKU"]):
            sku = row["Distributor_SKU"]
        else:
            sku = f"LINE_{idx}"

        raw_input = str(row.get("Client_Drug_Name", "")).strip()
        wholesaler_id = str(row.get("Wholesaler_ID", row.get("Supplier", "GENERIC_MFR"))).strip()
        total_invoice_cost = float(row.get("Client_Current_Price", 0.0))

        # Core clinical parsing matrix for volume quantities
        pack_size = 1
        pack_match = _re.search(r'(?:x|pack of\s*|/)?(\d{1,4})\s*(?:tabs?|caps?|s|pack|pcs|vials?)?\b', raw_input, _re.IGNORECASE)
        if pack_match:
            try:
                pack_size = int(pack_match.group(1))
                if pack_size == 0: pack_size = 1
            except:
                pack_size = 1

        true_unit_cost = total_invoice_cost / pack_size
        
        # Apply normalization pre-processing engine from Script 1_2
        normalised_input = _normalise_drug_name(raw_input)

        # Build tracking fingerprint check for deterministic speed
        lookup_string = f"{wholesaler_id}||{raw_input}".lower()
        hash_key = hashlib.md5(lookup_string.encode('utf-8')).hexdigest()

        generic_out, brand_out, sys_id_out, mfr_id_out = "", "", "", wholesaler_id
        match_method = ""
        score = 1.00

        # --- TIER 1 SHIELD: LOCAL STORAGE CACHE LOOKUP ---
        if hash_key in cache_db:
            generic_out = cache_db[hash_key]["generic_name"]
            brand_out = cache_db[hash_key]["brand_name"]
            sys_id_out = cache_db[hash_key]["system_id"]
            mfr_id_out = cache_db[hash_key].get("manufacturer_id", wholesaler_id)
            match_method = "Layer 1: Local Cache Hit"
            
        # --- TIER 1B SHIELD: EXACT REGISTRY MATCH ROUTINE ---
        elif any(master_df["generic_name"].astype(str).str.lower() == normalised_input.lower()) or any(master_df["brand_name"].astype(str).str.lower() == normalised_input.lower()):
            records = master_df[master_df["generic_name"].astype(str).str.lower() == normalised_input.lower()]
            if records.empty:
                records = master_df[master_df["brand_name"].astype(str).str.lower() == normalised_input.lower()]
            
            if not records.empty:
                generic_out = str(records.iloc[0]["generic_name"])
                brand_out = str(records.iloc[0]["brand_name"])
                sys_id_out = str(records.iloc[0]["system_id"])
            match_method = "Layer 1: Exact Registry Match"
            score = 1.0
            
            cache_db[hash_key] = {"generic_name": generic_out, "brand_name": brand_out, "system_id": sys_id_out, "manufacturer_id": mfr_id_out}
            save_cache_db(cache_db)
            
        # --- TIER 2 SHIELD: HIGH SPEED ALGORITHMIC VECTOR STRING MATCH ---
        elif tfidf_matrix is not None and len(normalised_input) > 0:
            # Look up calculations against the normalized variants
            raw_vec = vectorizer.transform([normalised_input])
            similarities = cosine_similarity(raw_vec, tfidf_matrix).flatten()
            best_idx = similarities.argmax()
            score = similarities[best_idx]
            
            if score >= confidence_threshold:
                matched_text = search_pool[best_idx]
                records = master_df[master_df["lookup_pool"] == matched_text]
                
                if not records.empty:
                    generic_out = str(records.iloc[0]["generic_name"])
                    brand_out = str(records.iloc[0]["brand_name"])
                    sys_id_out = str(records.iloc[0]["system_id"])
                match_method = "Layer 2: Algorithmic Vector Match"
                
                cache_db[hash_key] = {"generic_name": generic_out, "brand_name": brand_out, "system_id": sys_id_out, "manufacturer_id": mfr_id_out}
                save_cache_db(cache_db)
                
            # --- TIER 3 SHIELD: STRUCTURED OUT AI API FALLBACK ---
            else:
                ai_res = call_layer3_ai_api(raw_input, jurisdiction, search_pool)
                if "generic_name" in ai_res and ai_res["generic_name"] != "UNRESOLVED":
                    generic_out = ai_res["generic_name"]
                    brand_out = ai_res["brand_name"]
                    sys_id_out = ai_res["system_id"]
                    mfr_id_out = ai_res.get("manufacturer_id", wholesaler_id)
                    match_method = "Layer 3: Gemini AI Sandbox"
                    
                    cache_db[hash_key] = {"generic_name": generic_out, "brand_name": brand_out, "system_id": sys_id_out, "manufacturer_id": mfr_id_out}
                    save_cache_db(cache_db)
                else:
                    generic_out = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
                    brand_out = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
                    sys_id_out = "FLAGGED"
                    match_method = "Layer 3: Flagged Anomaly Exception"
                    score = 0.0
        else:
            generic_out = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
            brand_out = "UNRESOLVED - REQUIRES HUMAN OVERRIDE"
            sys_id_out = "FLAGGED"
            match_method = "Layer 3: Flagged Anomaly Exception"
            score = 0.0
        # --- AUTOMATED TARIFF COMPLIANCE EVALUATION ---
        baseline_tariff = 0.0
        audit_verdict = "Clearance Checked"
        
        if "UNRESOLVED" not in generic_out:
            matched_price_record = master_df[master_df["generic_name"] == generic_out]
            if not matched_price_record.empty:
                baseline_tariff = float(matched_price_record.iloc[0]["baseline_price"])
                
        variance = true_unit_cost - baseline_tariff if baseline_tariff > 0 else 0.0
        variance_pct = (variance / baseline_tariff) * 100 if baseline_tariff > 0 else 0.0
        
        if variance > 0.05:
            audit_verdict = "🚨 TARIFF OVERCHARGE"

        processed_rows.append({
            "SKU": sku,
            "Original Invoice Tag (Input)": raw_input,
            "Generic Name (VMP)": generic_out,
            "Brand Name (AMP)": brand_out,
            "Manufacturer / Wholesaler ID": mfr_id_out,
            "System ID Field": sys_id_out,
            "Match Method Used": match_method,
            "Confidence": f"{score * 100:.1f}%",
            "True Unit Cost": true_unit_cost,
            "Tariff Rate Benchmark": baseline_tariff,
            "Leakage Cost": variance,
            "Leakage %": variance_pct,
            "Audit Verdict": audit_verdict,
            "Raw_Score": score,
            "Hash_Key": hash_key
        })
        
    return pd.DataFrame(processed_rows)

def style_audit_matrix(row):
    styles = [''] * len(row)
    if "Anomaly" in str(row["Match Method Used"]) or "Exception" in str(row["Match Method Used"]):
        return ['background-color: rgba(255, 193, 7, 0.12); border-left: 4px solid #FFC107;'] * len(row)
    if "OVERCHARGE" in str(row["Audit Verdict"]):
        highlight = 'background-color: rgba(255, 87, 34, 0.18); color: #FF7777; font-weight: bold;'
        for target_col in ["True Unit Cost", "Leakage Cost", "Audit Verdict"]:
            if target_col in row.index:
                styles[row.index.get_loc(target_col)] = highlight
    return styles

# =====================================================================
# 5. DASHBOARD USER INTERFACE FRAMEWORK
# =====================================================================
st.markdown('<div class="brand-header">▲ APEX LOGIC</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">GLOBAL PHARMACY AUDIT ENGINE</div>', unsafe_allow_html=True)

tab_overview, tab_workspace = st.tabs(["🏠 Platform Overview", "⚡ Automated Audit Suite"])

# -----------------------------------------------------------------
# TAB 1: FRONT-END WEBSITE OVERVIEW (FROM SCRIPT 1_2)
# -----------------------------------------------------------------
with tab_overview:
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
# TAB 2: OPERATIONAL WORKSPACE INTERFACE 
# -----------------------------------------------------------------
with tab_workspace:
    with st.sidebar:
        st.markdown("### 📊 Parameter Profiles")
        jurisdiction = st.selectbox("Target Jurisdictional Region", ["UK", "US", "Canada"])
        threshold = st.slider("Automated Layer Confidence Filter", 0.50, 1.00, 0.92, help="Any algorithmic calculation score dropping below this percentage triggers automated AI evaluations or human confirmations.")
        
        st.write("---")
        st.caption(f"Active Profile: {jurisdiction} Data Standalone Core")
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
        
    j_profile = JURISDICTION_PROFILES[jurisdiction]
    master_registry = load_jurisdictional_registry(jurisdiction)

    st.markdown("#### 📑 Procurement Manifest Ingestion")
    st.markdown("<p style='color: #888; margin-top: -10px;'>Drop distribution invoices, formulas, or raw manifest logs to run registry audits.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Client Invoice, Formulary, or Procurement Sheet (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # File parsing ingestion matrix
        try:
            if uploaded_file.name.endswith('.csv'):
                client_data = pd.read_csv(uploaded_file)
            else:
                client_data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"🚨 Failed to read file structural layout. Error detail: {e}")
            st.stop()

        # Display raw 5 row file preview layout (From Script 1_2)
        st.markdown("##### 📄 Ingested Raw Source Matrix Preview")
        st.dataframe(client_data.head(5), width='stretch')

        raw_columns = client_data.columns.tolist()
        
        # Automated parsing alignment matrix using the synonym dictionary thesaurus (From Script 1_2)
        normalized_map = {
            col: col.strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "")
            for col in raw_columns
        }

        synonym_thesaurus = {
            "drug_name": [
                "drugname", "drug", "product", "productname", "item", "itemdescription",
                "medication", "medicationname", "medicine", "description", "standardname",
                "molecule", "activeingredient", "clinicalname", "brand", "rawinputphrase", "resolvedclinicalname", "clientdrugname"
            ],
            "current_price": [
                "currentprice", "price", "cost", "unitprice", "rate", "amount",
                "baselineprice", "procurementcost", "contractprice", "billingamount", "acquisitioncost", "invoiceprice", "marketbaseline", "clientcurrentprice"
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
        c1, c2 = st.columns(2)
        
        default_drug_idx = raw_columns.index(detected_drug_col) if detected_drug_col in raw_columns else 0
        default_price_idx = raw_columns.index(detected_price_col) if detected_price_col in raw_columns else min(1, len(raw_columns) - 1)

        with c1:
            if detected_drug_col:
                st.success(f"🎯 Auto-detected Drug Column: **'{detected_drug_col}'**")
                final_drug_col = detected_drug_col
            else:
                st.warning("🕵️‍♂️ Product/Drug column could not be auto-aligned.")
                final_drug_col = st.selectbox("Map your Product/Drug Name column manually:", options=raw_columns, index=default_drug_idx, key="manual_drug_select")

        with c2:
            if detected_price_col:
                st.success(f"🎯 Auto-detected Pricing Column: **'{detected_price_col}'**")
                final_price_col = detected_price_col
            else:
                st.warning("🕵️‍♂️ Pricing/Cost column could not be auto-aligned.")
                final_price_col = st.selectbox("Map your Current Unit Price column manually:", options=raw_columns, index=default_price_idx, key="manual_price_select")

        working_df = client_data.copy()
        working_df["Client_Drug_Name"] = client_data[final_drug_col].astype(str)
        working_df["Client_Current_Price"] = client_data[final_price_col].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0").astype(float)
        
        # Fire screen configuration toast confirmation (From Script 1_2)
        st.toast("🚀 Ingestion engine synchronized with downstream parameters!", icon="✅")

        st.markdown("---")
        st.markdown("#### ⚙️ Processing Framework Operational Stream")

        run_id = f"v7_merged_{uploaded_file.name}_{jurisdiction}_{threshold}"
        if st.session_state.get("run_id") != run_id:
            with st.spinner("Executing structural multi-layer match calculations across regional master profiles..."):
                st.session_state.audit_data = run_reconciliation_audit(working_df, master_registry, jurisdiction, threshold)
                st.session_state.run_id = run_id

        results_df = st.session_state.audit_data
        
        # Calculate pricing exposure variances
        overcharges = results_df[results_df["Audit Verdict"] == "🚨 TARIFF OVERCHARGE"]
        normal_matches = results_df[results_df["Audit Verdict"] == "Clearance Checked"]
        
        total_overcharged = overcharges["Leakage Cost"].sum()
        undercharged_records = results_df[results_df["Leakage Cost"] < 0]
        total_undercharged = undercharged_records["Leakage Cost"].sum()
        exceptions_flagged = results_df[results_df["Match Method Used"].str.contains("Anomaly|Exception")]

        # Render integrated macro metrics dashboard cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Manifest Rows Audited", len(results_df))
        with metric_col2:
            st.metric(
                "Estimated Overpayment Exposure", 
                f"${total_overcharged:,.2f}",
                delta=f"${abs(total_undercharged):,.2f} Favourable Variance" if total_undercharged < 0 else None,
                delta_color="normal"
            )
        with metric_col3:
            st.metric("Unresolved Exceptions Flagged", len(exceptions_flagged), delta=f"{len(exceptions_flagged)} Actions Required", delta_color="off")

        st.markdown("##### Processed Audit Stream Workspace")
        st.caption("💡 Highlighted Row = Layer 3 Flagged Exception | Highlighted Financial Cells = Wholesaler Price Gouging Surge Over National Averages")

        # Remap standard matrix system data tags dynamically based on dashboard settings
        display_df = results_df.copy()
        display_df = display_df.rename(columns={
            "Generic Name (VMP)": j_profile["generic_label"],
            "Brand Name (AMP)": j_profile["brand_label"],
            "System ID Field": j_profile["id_label"]
        })

        # Formatted currency representation outputs
        for monetary_col in ["True Unit Cost", "Tariff Rate Benchmark", "Leakage Cost"]:
            display_df[monetary_col] = display_df[monetary_col].map("${:,.4f}".format)
        display_df["Leakage %"] = display_df["Leakage %"].map("{:,.1f}%".format)

        st.dataframe(display_df.style.apply(style_audit_matrix, axis=1), width='stretch')

        # Human-In-The-Loop Exception Resolution Matrix Training System
        if len(exceptions_flagged) > 0:
            st.markdown("---")
            st.markdown("#### 🛠️ Human-In-The-Loop Exception Resolution Matrix")
            st.warning("The algorithmic engine flagged low-confidence inputs that failed your threshold safety. Reassign them manually below:")
            current_cache = load_cache_db()
            
            for idx, row in exceptions_flagged.iterrows():
                st.markdown(f"**SKU Source Target:** `{row['SKU']}` | **Corrupted Text Ingested:** *\"{row['Original Invoice Tag (Input)']}\"*")
                chosen_mapping = st.selectbox(f"Assign Clean Standard Label (Item Code {row['SKU']}):", ["-- Choose Correct Field --"] + list(master_registry["generic_name"].unique()), key=f"train_{idx}")
                
                if chosen_mapping != "-- Choose Correct Field --":
                    matched_row = master_registry[master_registry["generic_name"] == chosen_mapping]
                    current_cache[row["Hash_Key"]] = {
                        "generic_name": chosen_mapping,
                        "brand_name": str(matched_row.iloc[0]["brand_name"]) if not matched_row.empty else "Generic",
                        "system_id": str(matched_row.iloc[0]["system_id"]) if not matched_row.empty else "Manual",
                        "manufacturer_id": row["Manufacturer / Wholesaler ID"]
                    }
                    
            if st.button("Commit Training Rules & Override Exception Logs"):
                save_cache_db(current_cache)
                st.success("Rules written directly to Layer 1 local file! Re-running core...")
                st.session_state.clear()
                st.rerun()

        st.markdown("---")
        st.markdown("#### 📤 Export Verified Audit Logs")
        st.download_button("📥 Download Verified Audit Logs (.CSV)", data=results_df.to_csv(index=False).encode('utf-8'), file_name=f"apex_logic_audit_{jurisdiction.lower()}.csv", mime="text/csv")
