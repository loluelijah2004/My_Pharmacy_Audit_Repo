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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

JURISDICTION_PROFILES = {
    "UK":     {"csv_key": "UK",     "data_dir": "uk", "generic_label": "VMP Name",            "brand_label": "AMP Name",          "id_label": "VMP/AMP ID"},
    "US":     {"csv_key": "US",     "data_dir": "us", "generic_label": "Established Name",    "brand_label": "Proprietary Name",  "id_label": "NDC Number"},
    "Canada": {"csv_key": "CANADA", "data_dir": "ca", "generic_label": "Active Ingredient",   "brand_label": "Brand Name",        "id_label": "DIN"},
}

# Regional synonym map — invoice abbreviations often use UK names on US invoices.
REGIONAL_SYNONYMS = {
    "US":     {"paracetamol": "acetaminophen"},
    "UK":     {"acetaminophen": "paracetamol"},
    "CANADA": {"paracetamol": "acetaminophen"},
}

# =====================================================================
# 2. CACHE DATABASE (Layer 1 persistent store)
# =====================================================================
CACHE_FILE = os.path.join(BASE_PATH, "layer1_cache_db.json")
ABBREV_FILE = os.path.join(BASE_PATH, "abbreviations.json")

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
# 3. NOISE STRIPPING (used by registry normalisation + Layer 1)
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
    name = re.sub(r'\b[a-zA-Z]\b', '', name).strip()
    return " ".join(name.split())


# =====================================================================
# 4. MASTER REGISTRY LOADER
#    master_registry.csv has baseline prices but omits many common drugs.
#    Regional FDA / NHS / DPD files supplement the searchable drug list.
# =====================================================================
_COL_RENAME = {
    "standard_name": "generic_name", "drug_name": "generic_name",
    "active_ingredient": "generic_name", "vmp_name": "generic_name",
    "amp_name": "brand_name", "proprietary_name": "brand_name", "product_name": "brand_name",
    "regional_baseline_price": "baseline_price", "price": "baseline_price",
    "unit_price": "baseline_price", "cost": "baseline_price",
    "country": "region", "jurisdiction": "region", "market": "region",
    "ndc": "system_id", "din": "system_id",
}

_SALT_SUFFIX = re.compile(
    r'\s+(?:hydrochloride|hcl|hci|sodium|potassium|maleate|tartrate|besylate|'
    r'fumarate|succinate|acetate|sulfate|sulphate|phosphate|nitrate|bromide|'
    r'mononitrate|dihydrate|anhydrous|mesylate|lactate|citrate)\b.*$',
    re.IGNORECASE
)


def _normalize_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip().str.upper()
    for col in ["generic_name", "brand_name", "baseline_price", "system_id"]:
        if col not in df.columns:
            df[col] = "N/A"
    return df


def _base_ingredient(name: str) -> str:
    """Reduce a registry label to a matchable ingredient token."""
    text = strip_noise(str(name))
    text = _SALT_SUFFIX.sub("", text).strip()
    text = re.split(r"[/+]", text)[0].strip()
    return " ".join(w.capitalize() for w in text.split())


@st.cache_data
def _load_full_registry() -> pd.DataFrame:
    csv_path = os.path.join(BASE_PATH, "master_registry.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return _normalize_registry_columns(pd.read_csv(csv_path))
    except Exception:
        return pd.DataFrame()


@st.cache_data
def _load_regional_supplement(region_key: str, data_dir: str) -> pd.DataFrame:
    """Load FDA / NHS / DPD regional file to fill gaps in master_registry."""
    path = os.path.join(BASE_PATH, "data", data_dir, f"master_{data_dir}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = _normalize_registry_columns(pd.read_csv(path))
        df["region"] = region_key
        base_map = {n: _base_ingredient(n) for n in df["generic_name"].dropna().unique()}
        df["_base"] = df["generic_name"].map(base_map)
        df = df.dropna(subset=["_base"])
        df = df[df["_base"].str.len() >= 3]
        df["generic_name"] = df["_base"]
        df = df.drop_duplicates(subset=["generic_name"], keep="first")
        return df.drop(columns=["_base"])
    except Exception:
        return pd.DataFrame()


def load_jurisdictional_registry(region: str) -> pd.DataFrame:
    profile    = JURISDICTION_PROFILES.get(region, {})
    target_key = profile.get("csv_key", region.upper())
    data_dir   = profile.get("data_dir", region.lower())

    base_df = _load_full_registry()
    base_reg = (
        base_df[base_df["region"] == target_key].copy()
        if not base_df.empty else
        pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id", "region"])
    )

    supplement = _load_regional_supplement(target_key, data_dir)
    if supplement.empty and base_reg.empty:
        st.sidebar.error("❌ No registry data found for this region.")
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id"])

    if base_reg.empty:
        merged = supplement
    elif supplement.empty:
        merged = base_reg
    else:
        # Prefer master_registry baseline prices; add regional drugs not in baseline file.
        base_names = set(base_reg["generic_name"].str.lower())
        extra = supplement[~supplement["generic_name"].str.lower().isin(base_names)]
        merged = pd.concat([base_reg, extra], ignore_index=True)

    merged = merged.drop_duplicates(subset=["generic_name"], keep="first").reset_index(drop=True)
    st.sidebar.success(
        f"✅ {len(merged):,} drugs loaded for {region} "
        f"({len(base_reg):,} priced + {max(0, len(merged) - len(base_reg)):,} regional)."
    )
    return merged


@st.cache_data
def build_registry_index(_registry_key: str, generic_names: tuple) -> dict[str, int]:
    """Maps normalized ingredient tokens to row indices for O(1) Layer 1 resolution."""
    index: dict[str, int] = {}
    synonyms = REGIONAL_SYNONYMS.get(_registry_key, {})

    for i, generic in enumerate(generic_names):
        generic = str(generic)
        keys = {generic.lower(), _base_ingredient(generic).lower()}
        for key in keys:
            if not key:
                continue
            if key not in index or len(generic) < len(str(generic_names[index[key]])):
                index[key] = i
            alt = synonyms.get(key)
            if alt and (alt not in index or len(generic) < len(str(generic_names[index[alt]]))):
                index[alt] = i
        # First-token index: prefer the shortest matching registry name
        first = generic.lower().split()[0] if generic.split() else ""
        if first and (first not in index or len(generic) < len(str(generic_names[index[first]]))):
            index[first] = i
    return index


def resolve_in_registry(
    candidate: str, master_df: pd.DataFrame, registry_index: dict[str, int], region_key: str
) -> tuple[str | None, int | None]:
    """Map an abbreviation or cleaned token to the best registry row."""
    if not candidate or master_df.empty:
        return None, None

    synonyms = REGIONAL_SYNONYMS.get(region_key, {})
    probes = [
        candidate.lower().strip(),
        _base_ingredient(candidate).lower(),
        synonyms.get(candidate.lower().strip(), ""),
    ]
    first_word = candidate.lower().strip().split()[0] if candidate.split() else ""
    if first_word:
        probes.append(first_word)

    for probe in probes:
        if probe and probe in registry_index:
            idx = registry_index[probe]
            return str(master_df.iloc[idx]["generic_name"]), idx

    # Prefix: registry name extends the probe (e.g. "lisinopril" → "lisinopril hydrochloride")
    best_idx, best_len = None, 0
    for probe in probes:
        if not probe or len(probe) < 4:
            continue
        for key, idx in registry_index.items():
            if key.startswith(probe) and len(key) > best_len:
                best_idx, best_len = idx, len(key)
    if best_idx is not None:
        return str(master_df.iloc[best_idx]["generic_name"]), best_idx
    return None, None


# =====================================================================
# 5. LAYER 1: DETERMINISTIC ABBREVIATION LOOKUP
#    Loads custom abbreviations from an external abbreviations.json file.
# =====================================================================

@st.cache_data
def _load_abbrev_dict() -> dict:
    try:
        with open(ABBREV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Wholesaler shorthands that collide with obscure registry ingredient names.
_PHARMA_OVERRIDES = {
    "gaba": "Gabapentin", "gab": "Gabapentin",
    "para": "Paracetamol", "pcm": "Paracetamol",
    "met": "Metformin", "amox": "Amoxicillin",
    "lisino": "Lisinopril", "atorva": "Atorvastatin",
    "simva": "Simvastatin", "ibu": "Ibuprofen", "amlo": "Amlodipine",
}


@st.cache_data
def _build_abbrev_lookup(registry_key: str, registry_names: tuple) -> tuple:
    """Merge JSON abbreviations with auto-generated registry prefixes."""
    combined = dict(_load_abbrev_dict())
    for name in registry_names:
        clean = str(name).lower().strip()
        if len(clean) <= 6 or "(" in clean or len(clean) > 40:
            continue
        for n in (4, 5, 6):
            combined.setdefault(clean[:n], str(name))
    combined.update(_PHARMA_OVERRIDES)
    # Longest keys first for greedy prefix matching
    sorted_items = tuple(sorted(combined.items(), key=lambda x: len(x[0]), reverse=True))
    return sorted_items


def abbrev_lookup(clean_token: str, abbrev_items: tuple) -> str | None:
    """Returns canonical name if token matches a known abbreviation, else None."""
    t = clean_token.lower().strip()
    for abbrev, full_name in abbrev_items:
        if re.match(r'^' + re.escape(abbrev.lower().strip()) + r'(\s|/|$)', t, re.IGNORECASE):
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


def tfidf_match_batch(
    clean_tokens: list[str],
    master_df: pd.DataFrame,
    region_key: str,
    threshold: float,
) -> list[tuple[str | None, float]]:
    """Batch TF-IDF match — one matrix multiply for all unresolved rows."""
    n = len(clean_tokens)
    if master_df.empty or n == 0:
        return [(None, 0.0)] * n

    pool_list = (
        master_df["generic_name"].astype(str) + " " + master_df["brand_name"].astype(str)
    ).str.strip().tolist()

    try:
        vectorizer, matrix = build_tfidf_index(region_key, tuple(pool_list))
        results: list[tuple[str | None, float]] = []
        chunk_size = 2000
        for start in range(0, n, chunk_size):
            queries = [t.lower() for t in clean_tokens[start:start + chunk_size]]
            score_matrix = cosine_similarity(vectorizer.transform(queries), matrix)
            for row_scores in score_matrix:
                best_idx   = int(np.argmax(row_scores))
                best_score = float(row_scores[best_idx])
                if best_score >= threshold:
                    results.append((str(master_df.iloc[best_idx]["generic_name"]), best_score))
                else:
                    results.append((None, best_score))
        return results
    except Exception:
        return [(None, 0.0)] * n


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
        if result.get("resolved_name"):
            validated = result["resolved_name"]
            if validated not in registry_names:
                # Accept case-insensitive or synonym-normalised registry hits
                lower_map = {n.lower(): n for n in registry_names}
                validated = lower_map.get(validated.lower())
            if not validated:
                result["resolved_name"] = None
                result["confidence"]    = 0.0
            else:
                result["resolved_name"] = validated
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
      Phase 5  — TF-IDF cosine similarity (Layer 2, batched)
      Phase 6  — Gemini AI fallback (Layer 3)
      Phase 7  — Audit: compare invoice price vs registry baseline
    """
    n_rows = len(client_df)
    if n_rows == 0 or master_df.empty:
        return pd.DataFrame()

    region_key      = JURISDICTION_PROFILES.get(jurisdiction, {}).get("csv_key", jurisdiction)
    registry_index  = build_registry_index(region_key, tuple(master_df["generic_name"].tolist()))
    registry_names  = master_df["generic_name"].tolist()
    abbrev_items    = _build_abbrev_lookup(region_key, tuple(registry_names))

    cache_db = load_cache_db()
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}

    # Pre-allocate per-row state (avoids repeated DataFrame scans)
    skus           = [""] * n_rows
    raw_inputs     = [""] * n_rows
    clean_tokens   = [""] * n_rows
    hash_keys      = [""] * n_rows
    wholesaler_ids = [""] * n_rows
    invoice_prices = [0.0] * n_rows
    resolved       = [None] * n_rows
    match_layers   = [""] * n_rows
    confidences    = [0.0] * n_rows
    brand_names    = ["N/A"] * n_rows
    system_ids     = ["N/A"] * n_rows
    cache_dirty    = False

    # ── Pass 1: ingest + Layer 1 (cache, abbrev, exact) ──────────────
    for i, row in enumerate(client_df.itertuples(index=True)):
        idx = row.Index
        skus[i] = (
            str(getattr(row, "SKU", ""))
            if hasattr(row, "SKU") and pd.notna(getattr(row, "SKU", None))
            else str(getattr(row, "Distributor_SKU", f"LINE_{idx}"))
            if hasattr(row, "Distributor_SKU") and pd.notna(getattr(row, "Distributor_SKU", None))
            else f"LINE_{idx}"
        )
        raw_inputs[i]     = str(getattr(row, "Client_Drug_Name", "")).strip()
        wholesaler_ids[i] = str(getattr(row, "Wholesaler_ID", getattr(row, "Supplier", "GENERIC_MFR"))).strip()
        invoice_prices[i] = float(getattr(row, "Client_Current_Price", 0.0))
        clean_tokens[i]   = strip_noise(raw_inputs[i])
        hash_keys[i]      = hashlib.md5(f"{wholesaler_ids[i]}||{raw_inputs[i]}".lower().encode()).hexdigest()

        cached = cache_db.get(hash_keys[i]) or st.session_state["ai_cache"].get(hash_keys[i])
        if cached:
            resolved[i]     = cached.get("generic_name", "")
            brand_names[i]  = cached.get("brand_name", "N/A")
            system_ids[i]   = cached.get("system_id", "N/A")
            match_layers[i] = "Layer 1: Cache Hit"
            confidences[i]  = 1.0
            continue

        abbrev_result = abbrev_lookup(clean_tokens[i], abbrev_items)
        if abbrev_result:
            name, reg_idx = resolve_in_registry(abbrev_result, master_df, registry_index, region_key)
            if name:
                resolved[i]     = name
                match_layers[i] = "Layer 1: Abbreviation Lookup"
                confidences[i]  = 1.0
                brand_names[i]  = str(master_df.iloc[reg_idx]["brand_name"])
                system_ids[i]   = str(master_df.iloc[reg_idx]["system_id"])
                continue

        name, reg_idx = resolve_in_registry(clean_tokens[i], master_df, registry_index, region_key)
        if name:
            resolved[i]     = name
            match_layers[i] = "Layer 1: Exact Name Match"
            confidences[i]  = 1.0
            brand_names[i]  = str(master_df.iloc[reg_idx]["brand_name"])
            system_ids[i]   = str(master_df.iloc[reg_idx]["system_id"])

    # ── Pass 2: Layer 2 TF-IDF (batched) ─────────────────────────────
    l2_pending = [i for i in range(n_rows) if not resolved[i]]
    if l2_pending:
        batch_tokens = [clean_tokens[i] for i in l2_pending]
        batch_hits   = tfidf_match_batch(batch_tokens, master_df, region_key, l2_threshold)
        for i, (tfidf_name, tfidf_score) in zip(l2_pending, batch_hits):
            if tfidf_name:
                reg_row = master_df[master_df["generic_name"] == tfidf_name].iloc[0]
                resolved[i]     = tfidf_name
                match_layers[i] = "Layer 2: TF-IDF Cosine Match"
                confidences[i]  = tfidf_score
                brand_names[i]  = str(reg_row["brand_name"])
                system_ids[i]   = str(reg_row["system_id"])
                cache_db[hash_keys[i]] = {
                    "generic_name": tfidf_name,
                    "brand_name":   brand_names[i],
                    "system_id":    system_ids[i],
                }
                cache_dirty = True

    # ── Pass 3: Layer 3 AI (only unresolved) ─────────────────────────
    if use_ai:
        for i in range(n_rows):
            if resolved[i]:
                continue
            ai_result = call_gemini_api(raw_inputs[i], clean_tokens[i], registry_names, jurisdiction)
            if ai_result.get("resolved_name"):
                ai_name = ai_result["resolved_name"]
                reg_row = master_df[master_df["generic_name"] == ai_name]
                resolved[i]     = ai_name
                match_layers[i] = "Layer 3: AI Resolution"
                confidences[i]  = float(ai_result.get("confidence", 0.0))
                brand_names[i]  = str(reg_row.iloc[0]["brand_name"]) if not reg_row.empty else "N/A"
                system_ids[i]   = str(reg_row.iloc[0]["system_id"])  if not reg_row.empty else "N/A"
                entry = {
                    "generic_name": ai_name,
                    "brand_name":   brand_names[i],
                    "system_id":    system_ids[i],
                }
                cache_db[hash_keys[i]] = entry
                st.session_state["ai_cache"][hash_keys[i]] = entry
                cache_dirty = True

    if cache_dirty:
        save_cache_db(cache_db)

    # ── Pass 4: flag unresolved + price audit ───────────────────────
    rows = []
    for i in range(n_rows):
        resolved_name = resolved[i]
        if not resolved_name:
            resolved_name = "UNRESOLVED — HUMAN OVERRIDE REQUIRED"
            match_layers[i] = "Layer 3: Flagged Exception"
            confidences[i]  = 0.0
            baseline_price  = 0.0
        else:
            reg_row = master_df[master_df["generic_name"] == resolved_name]
            if not reg_row.empty:
                brand_names[i]  = str(reg_row.iloc[0]["brand_name"])
                system_ids[i]   = str(reg_row.iloc[0]["system_id"])
                baseline_price  = float(reg_row.iloc[0]["baseline_price"])
            else:
                baseline_price = 0.0

        variance     = (invoice_prices[i] - baseline_price) if baseline_price > 0 else 0.0
        variance_pct = (variance / baseline_price * 100) if baseline_price > 0 else 0.0
        verdict      = "🚨 TARIFF OVERCHARGE" if variance > 0.05 else "✅ Clearance Checked"

        rows.append({
            "SKU":                  skus[i],
            "Original Invoice Tag": raw_inputs[i],
            "Cleaned Token":        clean_tokens[i],
            "Generic Name":         resolved_name,
            "Brand Name":           brand_names[i],
            "Wholesaler ID":        wholesaler_ids[i],
            "System ID":            system_ids[i],
            "Match Layer":          match_layers[i],
            "Confidence":           f"{confidences[i] * 100:.1f}%",
            "Invoice Price":        invoice_prices[i],
            "Registry Baseline":    baseline_price,
            "Price Variance":       variance,
            "Variance %":           variance_pct,
            "Audit Verdict":        verdict,
            "_raw_score":           confidences[i],
            "_hash":                hash_keys[i],
        })

    return pd.DataFrame(rows)


# =====================================================================
# 9. DISPLAY HELPER
# =====================================================================
def style_rows(row):
    # Use .get() to safely check for multiple possible column names
    match_layer = str(row.get("Match Layer", ""))
    generic_name = str(row.get("Generic Name", row.get("generic_name", row.get("VMP Name", ""))))
    
    if "Flagged" in match_layer or "UNRESOLVED" in generic_name:
        return ['background-color: #ffcccc'] * len(row)
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
        sku_col = next(
            (c for c in raw_columns
             if normalized_map[c] in ("sku", "vendorsku", "distributorsku", "productcode", "itemcode")),
            None
        )
        if sku_col:
            working_df["SKU"] = client_data[sku_col].astype(str)
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
