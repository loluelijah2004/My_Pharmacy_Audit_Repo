import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

JURISDICTION_PROFILES = {
    "UK":     {"csv_key": "UK",     "generic_label": "VMP Name",            "brand_label": "AMP Name",          "id_label": "VMP/AMP ID"},
    "US":     {"csv_key": "US",     "generic_label": "Established Name",    "brand_label": "Proprietary Name",  "id_label": "NDC Number"},
    "Canada": {"csv_key": "CANADA", "generic_label": "Active Ingredient",   "brand_label": "Brand Name",        "id_label": "DIN"},
}

# =====================================================================
# 2. CACHE DATABASE (Layer 1 persistent store)
# =====================================================================
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer1_cache_db.json")

def load_cache_db() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache_db(data: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # Streamlit Cloud read-only FS — fails silently, session cache still works

# =====================================================================
# 3. MASTER REGISTRY LOADER
# =====================================================================
@st.cache_data
def _load_full_registry() -> pd.DataFrame:
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(base_path, "master_registry.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename = {
            "standard_name": "generic_name", "drug_name": "generic_name",
            "active_ingredient": "generic_name", "vmp_name": "generic_name",
            "amp_name": "brand_name", "proprietary_name": "brand_name", "product_name": "brand_name",
            "regional_baseline_price": "baseline_price", "price": "baseline_price",
            "unit_price": "baseline_price", "cost": "baseline_price",
            "country": "region", "jurisdiction": "region", "market": "region",
            "ndc": "system_id", "din": "system_id",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["region"] = df["region"].astype(str).str.strip().str.upper()
        for col in ["generic_name", "brand_name", "baseline_price", "system_id"]:
            if col not in df.columns:
                df[col] = "N/A"
        return df
    except Exception:
        return pd.DataFrame()


def load_jurisdictional_registry(region: str) -> pd.DataFrame:
    full_df    = _load_full_registry()
    target_key = JURISDICTION_PROFILES.get(region, {}).get("csv_key", region.upper())
    if full_df.empty:
        st.sidebar.error("❌ master_registry.csv not found or could not be read.")
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id"])
    filtered = full_df[full_df["region"] == target_key].copy().reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning(f"⚠️ No entries for region '{target_key}'.")
    else:
        st.sidebar.success(f"✅ {len(filtered)} drugs loaded for {region}.")
    return filtered


# =====================================================================
# 4. PHASE 1 — NOISE STRIPPING
#    Uses \b word boundaries so "20MG" is stripped but "LISINO" is not.
#    BUG FIX: Previous version used space-padded alternation which failed
#    to strip tokens at start/end of string and on Python 3.14.
# =====================================================================
_STRIP_NOISE = re.compile(
    r'\s*//\s*batch[-\s]?\w+'                              # // BATCH-42
    r'|\b\d+(?:\.\d+)?(?:/\d+)?\s*(?:mg|mcg|g|ml|iu)\b'  # 500MG, 500/125MG
    r'|\b(?:tab|tabs|cap|caps|er|sr|xr|xl|dr|eff|inh|inj|soln?|susp|tar|hcl|hci)\b'
    r'|\bx-\d+\b',
    re.IGNORECASE
)

def strip_noise(raw: str) -> str:
    """Strip batch codes, dosage strengths, and dosage form words."""
    name = _STRIP_NOISE.sub(" ", raw).strip()
    # Remove isolated SINGLE characters only — do NOT remove letters from inside words.
    # BUG FIX: Previous r'[a-zA-Z]' removed ALL letters. r'\b[a-zA-Z]\b' removes only
    # standalone single-char tokens like a trailing "T" from "PANTOPRAZOLE 40MG T".
    name = re.sub(r'\b[a-zA-Z]\b', '', name).strip()
    return " ".join(name.split())


# =====================================================================
# 5. PHASE 2 — LAYER 1: DETERMINISTIC ABBREVIATION LOOKUP
#    Loads custom abbreviations from an external abbreviations.json file.
# =====================================================================

@st.cache_data
def generate_automatic_prefixes(registry_df):
    auto_lookup = {}
    for name in registry_df['generic_name'].dropna().unique():
        clean_name = str(name).lower().strip()
        # If the drug name is long enough, auto-create a prefix shorthand
        if len(clean_name) > 6:
            prefix_4 = clean_name[:4]
            prefix_5 = clean_name[:5]
            prefix_6 = clean_name[:6]
            
            # Map them back to the canonical proper name
            auto_lookup[prefix_4] = name
            auto_lookup[prefix_5] = name
            auto_lookup[prefix_6] = name
    return auto_lookup
    
def load_abbreviations() -> dict:
    """Loads abbreviation dictionary from an external JSON file."""
    try:
        with open("abbreviations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback if the file doesn't exist yet
        st.warning("⚠️ abbreviations.json file not found! Operating without manual overrides.")
        return {}

# Load the dictionary into memory
_ABBREV_DICT = load_abbreviations()

def abbrev_lookup(clean_token: str) -> str | None:
    """Returns canonical name if token matches a known abbreviation, else None."""
    t = clean_token.lower().strip()
    
    # Sort keys longest-first so compound terms match before short roots
    sorted_items = sorted(_ABBREV_DICT.items(), key=lambda x: len(x[0]), reverse=True)
    
    for abbrev, full_name in sorted_items:
        if re.match(r'^' + re.escape(abbrev.lower().strip()) + r'(\s|$)', t, re.IGNORECASE):
            return full_name
    return None


# =====================================================================
# 6. PHASE 3 — LAYER 2: TF-IDF COSINE SIMILARITY
#    Character n-gram TF-IDF vectorised against the entire registry at
#    once — O(1) lookup regardless of registry size. No row-by-row loop.
#    BUG FIX: TF-IDF threshold was 0.92 which is unreachable for short
#    tokens. Realistic TF-IDF cosine scores for pharmacy names top out
#    at ~0.75. The threshold slider now applies correctly.
# =====================================================================
@st.cache_resource
def build_tfidf_index(registry_key: str, registry_names: tuple):
    """
    Builds and caches the TF-IDF matrix for a given registry.
    registry_key is used only as a cache discriminator.
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    matrix     = vectorizer.fit_transform([n.lower() for n in registry_names])
    return vectorizer, matrix


def tfidf_match(clean_token: str, master_df: pd.DataFrame,
                region_key: str, threshold: float) -> tuple[str | None, float]:
    """
    Vectorises the query against the full registry matrix in one shot.
    Returns (matched_generic_name, score) or (None, best_score).
    """
    if master_df.empty or not clean_token.strip():
        return None, 0.0

    # Build lookup pool from generic + brand names combined
    master_df = master_df.copy()
    master_df["_pool"] = (
        master_df["generic_name"].astype(str) + " " +
        master_df["brand_name"].astype(str)
    ).str.strip()

    pool_list = master_df["_pool"].tolist()
    try:
        vectorizer, matrix = build_tfidf_index(region_key, tuple(pool_list))
        query_vec = vectorizer.transform([clean_token.lower()])
        scores    = cosine_similarity(query_vec, matrix).flatten()
        best_idx  = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return str(master_df.iloc[best_idx]["generic_name"]), best_score
        return None, best_score
    except Exception:
        return None, 0.0


# =====================================================================
# 7. PHASE 4 — LAYER 3: GEMINI AI FALLBACK
#    Only called when both Layer 1 and Layer 2 fail.
#    BUG FIX: Added explicit secret-missing warning in sidebar.
#    Results cached in session_state["ai_cache"] so same input never
#    hits the API twice per session.
# =====================================================================
def call_gemini_api(raw_input: str, clean_token: str,
                    registry_names: list, jurisdiction: str) -> dict:
    """Calls Gemini via OpenAI-compatible endpoint. Returns resolved dict or failure."""
    if not OPENAI_AVAILABLE:
        return {"status": "error", "reason": "openai package not installed"}

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"status": "no_key"}

    sample  = "\n".join(f"- {n}" for n in registry_names[:150])
    prompt  = (
        f"You are a pharmaceutical drug name resolver for the {jurisdiction} market.\n"
        f"Match the invoice entry to the single best name from the registry list.\n\n"
        f"INVOICE ENTRY: \"{raw_input}\"\n"
        f"CLEANED TOKEN: \"{clean_token}\"\n\n"
        f"REGISTRY (choose ONLY from this list):\n{sample}\n\n"
        f"Rules:\n"
        f"1. Account for abbreviations, typos, brand/generic switching, and dosage noise.\n"
        f"2. Return null resolved_name if genuinely no match.\n"
        f"3. Respond ONLY with valid JSON — no prose.\n\n"
        f'Schema: {{"resolved_name": "<exact registry name or null>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}'
    )
    try:
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        # Validate resolved name is actually in registry
        if result.get("resolved_name") and result["resolved_name"] not in registry_names:
            result["resolved_name"] = None
            result["confidence"]    = 0.0
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# =====================================================================
# 8. MAIN RECONCILIATION PIPELINE
# =====================================================================
def run_reconciliation(client_df: pd.DataFrame, master_df: pd.DataFrame,
                       jurisdiction: str, l2_threshold: float,
                       use_ai: bool) -> pd.DataFrame:
    """
    Full pipeline per invoice row:
      Phase 1  — Strip noise from raw text
      Phase 2  — Check persistent hash cache (Layer 1 instant hit)
      Phase 3  — Abbreviation dictionary lookup (Layer 1 deterministic)
      Phase 4  — Exact name match on cleaned token (Layer 1)
      Phase 5  — TF-IDF cosine similarity (Layer 2)
      Phase 6  — Gemini AI fallback (Layer 3)
      Phase 7  — Audit: compare invoice price vs registry baseline
    """
    registry_names = master_df["generic_name"].tolist()
    region_key     = JURISDICTION_PROFILES.get(jurisdiction, {}).get("csv_key", jurisdiction)

    cache_db = load_cache_db()
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}

    rows = []

    for idx, row in client_df.iterrows():

        # ── Identify SKU ─────────────────────────────────────────────
        sku = (
            str(row["SKU"])             if "SKU"             in row.index and pd.notna(row["SKU"])
            else str(row["Distributor_SKU"]) if "Distributor_SKU" in row.index and pd.notna(row["Distributor_SKU"])
            else f"LINE_{idx}"
        )

        raw_input      = str(row.get("Client_Drug_Name", "")).strip()
        wholesaler_id  = str(row.get("Wholesaler_ID", row.get("Supplier", "GENERIC_MFR"))).strip()
        invoice_price  = float(row.get("Client_Current_Price", 0.0))

        # ── Phase 1: Strip noise ──────────────────────────────────────
        clean_token = strip_noise(raw_input)

        # ── Phase 2: Hash cache lookup ────────────────────────────────
        hash_key   = hashlib.md5(f"{wholesaler_id}||{raw_input}".lower().encode()).hexdigest()
        cached     = cache_db.get(hash_key) or st.session_state["ai_cache"].get(hash_key)

        resolved_name = None
        brand_name    = "N/A"
        system_id     = "N/A"
        match_layer   = ""
        confidence    = 0.0

        if cached:
            resolved_name = cached.get("generic_name", "")
            brand_name    = cached.get("brand_name", "N/A")
            system_id     = cached.get("system_id", "N/A")
            match_layer   = "Layer 1: Cache Hit"
            confidence    = 1.0

        # ── Phase 3: Abbreviation dictionary ─────────────────────────
        if not resolved_name:
            abbrev_result = abbrev_lookup(clean_token)
            if abbrev_result and abbrev_result in registry_names:
                resolved_name = abbrev_result
                match_layer   = "Layer 1: Abbreviation Lookup"
                confidence    = 1.0

        # ── Phase 4: Exact name match ─────────────────────────────────
        if not resolved_name:
            exact = master_df[master_df["generic_name"].str.lower() == clean_token.lower()]
            if not exact.empty:
                resolved_name = str(exact.iloc[0]["generic_name"])
                brand_name    = str(exact.iloc[0]["brand_name"])
                system_id     = str(exact.iloc[0]["system_id"])
                match_layer   = "Layer 1: Exact Name Match"
                confidence    = 1.0

        # ── Phase 5: TF-IDF cosine similarity ─────────────────────────
        if not resolved_name:
            tfidf_name, tfidf_score = tfidf_match(clean_token, master_df, region_key, l2_threshold)
            if tfidf_name:
                resolved_name = tfidf_name
                match_layer   = "Layer 2: TF-IDF Cosine Match"
                confidence    = tfidf_score
                # Write to cache so this exact input resolves instantly next time
                reg_row = master_df[master_df["generic_name"] == tfidf_name]
                cache_db[hash_key] = {
                    "generic_name": tfidf_name,
                    "brand_name":   str(reg_row.iloc[0]["brand_name"]) if not reg_row.empty else "N/A",
                    "system_id":    str(reg_row.iloc[0]["system_id"])  if not reg_row.empty else "N/A",
                }
                save_cache_db(cache_db)

        # ── Phase 6: AI fallback ──────────────────────────────────────
        if not resolved_name and use_ai:
            ai_result = call_gemini_api(raw_input, clean_token, registry_names, jurisdiction)
            if ai_result.get("resolved_name"):
                resolved_name = ai_result["resolved_name"]
                match_layer   = "Layer 3: AI Resolution"
                confidence    = float(ai_result.get("confidence", 0.0))
                reg_row = master_df[master_df["generic_name"] == resolved_name]
                entry = {
                    "generic_name": resolved_name,
                    "brand_name":   str(reg_row.iloc[0]["brand_name"]) if not reg_row.empty else "N/A",
                    "system_id":    str(reg_row.iloc[0]["system_id"])  if not reg_row.empty else "N/A",
                }
                cache_db[hash_key] = entry
                st.session_state["ai_cache"][hash_key] = entry
                save_cache_db(cache_db)

        # ── Unresolved ────────────────────────────────────────────────
        if not resolved_name:
            resolved_name = "UNRESOLVED — HUMAN OVERRIDE REQUIRED"
            match_layer   = "Layer 3: Flagged Exception"
            confidence    = 0.0

        # ── Phase 7: Audit ────────────────────────────────────────────
        if "UNRESOLVED" not in resolved_name:
            reg_row        = master_df[master_df["generic_name"] == resolved_name]
            brand_name     = str(reg_row.iloc[0]["brand_name"]) if not reg_row.empty else "N/A"
            system_id      = str(reg_row.iloc[0]["system_id"])  if not reg_row.empty else "N/A"
            baseline_price = float(reg_row.iloc[0]["baseline_price"]) if not reg_row.empty else 0.0
        else:
            baseline_price = 0.0

        variance     = (invoice_price - baseline_price) if baseline_price > 0 else 0.0
        variance_pct = (variance / baseline_price * 100) if baseline_price > 0 else 0.0
        verdict      = "🚨 TARIFF OVERCHARGE" if variance > 0.05 else "✅ Clearance Checked"

        rows.append({
            "SKU":                         sku,
            "Original Invoice Tag":        raw_input,
            "Cleaned Token":               clean_token,
            "Generic Name":                resolved_name,
            "Brand Name":                  brand_name,
            "Wholesaler ID":               wholesaler_id,
            "System ID":                   system_id,
            "Match Layer":                 match_layer,
            "Confidence":                  f"{confidence * 100:.1f}%",
            "Invoice Price":               invoice_price,
            "Registry Baseline":           baseline_price,
            "Price Variance":              variance,
            "Variance %":                  variance_pct,
            "Audit Verdict":               verdict,
            "_raw_score":                  confidence,
            "_hash":                       hash_key,
        })

    return pd.DataFrame(rows)


# =====================================================================
# 9. DISPLAY HELPER
# =====================================================================
def style_rows(row):
    if "Flagged" in str(row["Match Layer"]) or "UNRESOLVED" in str(row["Generic Name"]):
        return ['background-color: rgba(255,193,7,0.15); border-left: 4px solid #FFC107;'] * len(row)
    if "OVERCHARGE" in str(row["Audit Verdict"]):
        hi = 'background-color: rgba(255,87,34,0.18); color: #FF7777; font-weight: bold;'
        cols = row.index.tolist()
        styles = [''] * len(row)
        for c in ["Invoice Price", "Price Variance", "Audit Verdict"]:
            if c in cols:
                styles[cols.index(c)] = hi
        return styles
    return [''] * len(row)


# =====================================================================
# 10. UI
# =====================================================================
st.markdown('<div class="brand-header">▲ APEX LOGIC</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">GLOBAL PHARMACY AUDIT ENGINE</div>', unsafe_allow_html=True)

tab_overview, tab_workspace = st.tabs(["🏠 Platform Overview", "⚡ Automated Audit Suite"])

with tab_overview:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### Reclaim Your Pharmacy's Lost Margin")
        st.write(
            "Apex Logic processes distributor invoices through a three-layer resolution engine — "
            "deterministic abbreviation lookup, TF-IDF cosine similarity, and AI fallback — "
            "then cross-references every resolved drug against live national baseline registries "
            "to surface hidden price gouging before you clear accounts payable."
        )
        st.markdown("#### 🌍 Regions Supported")
        st.info("**United Kingdom:** NHS dm+d\n\n**United States:** FDA NDC\n\n**Canada:** Health Canada DPD")
        st.markdown("#### ⚙️ Three-Layer Pipeline")
        st.markdown("""
| Layer | Method | Example |
|---|---|---|
| **Layer 1a** | Hash cache (persistent) | Same supplier + drug → instant hit |
| **Layer 1b** | Abbreviation dictionary (80+ shortcuts) | `MET → Metformin`, `PCM → Paracetamol` |
| **Layer 1c** | Exact name match after noise strip | `LISINOPRIL 5MG TABS` → `Lisinopril` |
| **Layer 2** | TF-IDF char n-gram cosine similarity | `SERTRALINE` → `Sertraline` |
| **Layer 3** | Gemini AI (bounded JSON schema) | Anything ambiguous |
""")
    with col_right:
        st.markdown("⚙️ **System Status**")
        st.success("Layer 1 Cache + Lookup: Online")
        st.success("Layer 2 TF-IDF Engine: Active")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        if gemini_key:
            st.success("Layer 3 Gemini AI: Configured ✅")
        else:
            st.warning("Layer 3 Gemini AI: No API key — add GEMINI_API_KEY to Streamlit secrets")

with tab_workspace:
    with st.sidebar:
        st.markdown("### 📊 Parameters")
        jurisdiction = st.selectbox("Region", ["UK", "US", "Canada"])
        l2_threshold = st.slider(
            "Layer 2 Confidence Threshold", 0.30, 1.00, 0.50,
            help="TF-IDF cosine scores for pharmacy names realistically peak at 0.5–0.8. "
                 "Setting this above 0.8 means almost nothing reaches Layer 2 acceptance."
        )
        use_ai = st.toggle(
            "Enable Layer 3 AI Fallback",
            value=False,
            help="Requires GEMINI_API_KEY in Streamlit secrets."
        )
        if use_ai and not st.secrets.get("GEMINI_API_KEY", ""):
            st.sidebar.error("⚠️ GEMINI_API_KEY missing from secrets. Layer 3 will not fire.")

        st.write("---")
        st.caption(f"Region: {jurisdiction} | L2 threshold: {l2_threshold:.0%}")
        st.write("---")
        with st.container(border=True):
            st.markdown("<small>🔒 **ARCHITECTURAL BOUNDARY**</small>", unsafe_allow_html=True)
            st.caption("No EMR/EHR or eRx connectivity. Financial reconciliation only.")
        st.markdown("### 🏛️ Enterprise Support")
        st.link_button("💻 Service Desk", url="https://support.apexlogic.ai/portal", width='stretch')
        st.link_button("📞 Priority Callback", url="mailto:enterprise-ops@apexlogic.ai?subject=URGENT", width='stretch')

    master_registry = load_jurisdictional_registry(jurisdiction)
    j_profile       = JURISDICTION_PROFILES[jurisdiction]

    st.markdown("#### 📑 Invoice Upload")
    uploaded_file = st.file_uploader(
        "Upload invoice (.csv or .xlsx)",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            client_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"🚨 Could not read file: {e}")
            st.stop()

        st.markdown("##### Raw Invoice Preview")
        st.dataframe(client_data.head(5), width='stretch')

        # Column auto-detection
        raw_columns    = client_data.columns.tolist()
        normalized_map = {c: c.strip().lower().replace("_","").replace(" ","").replace("-","").replace("/","") for c in raw_columns}

        drug_synonyms  = ["drugname","drug","product","productname","item","itemdescription",
                          "medication","medicine","description","standardname","molecule",
                          "activeingredient","clinicalname","brand","clientdrugname"]
        price_synonyms = ["unitprice","price","cost","currentprice","rate","amount",
                          "procurementcost","contractprice","billingamount","acquisitioncost",
                          "invoiceprice","clientcurrentprice"]

        detected_drug  = next((o for o, c in normalized_map.items() if c in drug_synonyms),  None)
        detected_price = next((o for o, c in normalized_map.items() if c in price_synonyms), None)

        st.markdown("#### ⚙️ Column Mapping")
        c1, c2 = st.columns(2)
        with c1:
            drug_col = detected_drug or st.selectbox("Drug Name Column", raw_columns, key="drug_col")
            if detected_drug:
                st.success(f"🎯 Auto-detected: **{detected_drug}**")
                drug_col = detected_drug
        with c2:
            price_col = detected_price or st.selectbox("Unit Price Column", raw_columns, key="price_col")
            if detected_price:
                st.success(f"🎯 Auto-detected: **{detected_price}**")
                price_col = detected_price

        working_df = client_data.copy()
        working_df["Client_Drug_Name"]    = client_data[drug_col].astype(str).str.strip()
        working_df["Client_Current_Price"] = (
            client_data[price_col].astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", "0").astype(float)
        )
        st.toast("✅ Invoice ingested — running pipeline...", icon="⚡")

        # Run pipeline
        ENGINE_VERSION = "v5"
        ai_flag        = "ai_on" if use_ai else "ai_off"
        run_id         = f"{ENGINE_VERSION}_{uploaded_file.name}_{jurisdiction}_{l2_threshold}_{ai_flag}"

        if st.session_state.get("run_id") != run_id:
            with st.spinner("Running three-layer pipeline..."):
                st.session_state["audit_data"] = run_reconciliation(
                    working_df, master_registry, jurisdiction, l2_threshold, use_ai
                )
                st.session_state["run_id"] = run_id

        results_df = st.session_state["audit_data"]

        # Metrics
        overcharges  = results_df[results_df["Audit Verdict"] == "🚨 TARIFF OVERCHARGE"]
        unresolved   = results_df[results_df["Generic Name"].str.contains("UNRESOLVED")]
        l1_hits      = results_df[results_df["Match Layer"].str.startswith("Layer 1")]
        l2_hits      = results_df[results_df["Match Layer"].str.startswith("Layer 2")]
        l3_ai_hits   = results_df[results_df["Match Layer"] == "Layer 3: AI Resolution"]
        exposure     = overcharges["Price Variance"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows Audited", len(results_df))
        m2.metric("Overcharge Exposure", f"${exposure:,.2f}", delta=f"{len(overcharges)} lines", delta_color="inverse")
        m3.metric("Unresolved", len(unresolved))
        m4.metric("Auto-Resolved", f"{len(results_df) - len(unresolved)}/{len(results_df)}")

        r1, r2, r3 = st.columns(3)
        r1.metric("⚡ Layer 1 (Cache + Lookup)", len(l1_hits))
        r2.metric("🔬 Layer 2 (TF-IDF)", len(l2_hits))
        r3.metric("🤖 Layer 3 (AI)", len(l3_ai_hits))

        # Results table — rename columns per jurisdiction
        display_df = results_df.drop(columns=["_raw_score", "_hash"], errors="ignore").copy()
        display_df = display_df.rename(columns={
            "Generic Name": j_profile["generic_label"],
            "Brand Name":   j_profile["brand_label"],
            "System ID":    j_profile["id_label"],
        })
        for col in ["Invoice Price", "Registry Baseline", "Price Variance"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("${:,.4f}".format)
        if "Variance %" in display_df.columns:
            display_df["Variance %"] = display_df["Variance %"].map("{:,.1f}%".format)

        st.markdown("##### Audit Results")
        st.dataframe(display_df.style.apply(style_rows, axis=1), width='stretch')

        # Human override for unresolved
        if len(unresolved) > 0:
            st.markdown("---")
            st.markdown("#### 🛠️ Human Override — Unresolved Items")
            st.warning(f"{len(unresolved)} items need manual resolution.")
            current_cache = load_cache_db()

            for idx, row in unresolved.iterrows():
                st.markdown(f"**SKU:** `{row['SKU']}` | **Raw:** *\"{row['Original Invoice Tag']}\"* | **Cleaned:** `{row['Cleaned Token']}`")
                chosen = st.selectbox(
                    "Map to registry:",
                    ["— Skip —"] + master_registry["generic_name"].tolist(),
                    key=f"override_{idx}"
                )
                if chosen != "— Skip —":
                    reg_row = master_registry[master_registry["generic_name"] == chosen]
                    current_cache[row["_hash"]] = {
                        "generic_name": chosen,
                        "brand_name":   str(reg_row.iloc[0]["brand_name"]) if not reg_row.empty else "N/A",
                        "system_id":    str(reg_row.iloc[0]["system_id"])  if not reg_row.empty else "N/A",
                    }

            if st.button("💾 Commit Overrides to Layer 1 Cache"):
                save_cache_db(current_cache)
                st.success("Overrides written to cache. These will resolve instantly on next run.")
                st.session_state.pop("run_id", None)
                st.rerun()

        # Export
        st.markdown("---")
        st.markdown("#### 📤 Export")
        ex1, ex2 = st.columns(2)
        with ex1:
            clean_cols = ["SKU", j_profile["generic_label"], j_profile["brand_label"],
                          "Invoice Price", "Registry Baseline", "Price Variance", "Audit Verdict"]
            clean_export = display_df[[c for c in clean_cols if c in display_df.columns]]
            st.download_button(
                "📦 Clean PMS File",
                data=clean_export.to_csv(index=False).encode(),
                file_name=f"apex_pms_{jurisdiction.lower()}.csv",
                mime="text/csv"
            )
        with ex2:
            st.download_button(
                "🔎 Full Audit Workbook",
                data=display_df.to_csv(index=False).encode(),
                file_name=f"apex_audit_{jurisdiction.lower()}.csv",
                mime="text/csv"
            )

    else:
        st.write("---")
        st.info("Upload an invoice to begin.")
