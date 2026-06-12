import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import hashlib
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# =====================================================================
# 0. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# =====================================================================
st.set_page_config(
    page_title="Apex Logic | Professional Invoice Harmonization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session states for navigation and authentication
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "audit_data" not in st.session_state:
    st.session_state["audit_data"] = None
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None

# Safely extract Whop API Token from Streamlit Secrets
WHOP_API_KEY = st.secrets.get("WHOP_API_KEY", "")

# =====================================================================
# 0b. WHOP AUTHENTICATION FUNCTION
# =====================================================================
def check_whop_authorization(membership_id: str) -> bool:
    """Verifies membership directly with Whop's ledger. Locks out unauthorized shares."""
    
    # 🛠️ THE DEVELOPER BACKDOOR (Remove or change before public launch!)
    if membership_id == "apex_admin_2026":
        return True
        
    if not membership_id or not WHOP_API_KEY:
        return False
        
    url = f"https://api.whop.com/v5/memberships/{membership_id}"
    headers = {
        "Authorization": f"Bearer {WHOP_API_KEY}", 
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get("status") == "active"
    except Exception:
        return False
    return False

# =====================================================================
# 1. PLATFORM CONFIGURATION & JURISDICTIONAL TAXONOMY
# =====================================================================
st.markdown("""
    <style>
        .brand-header { font-size: 42px; font-weight: 800; letter-spacing: 2px; color: #FFFFFF; margin-bottom: 0px; }
        .brand-subtitle { font-size: 14px; font-weight: 400; letter-spacing: 4px; color: #00FFCC; margin-top: 0px; margin-bottom: 25px; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 12px 24px; }
        .hero-title { font-size: 3rem !important; font-weight: 800 !important; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.5rem; }
        .accent-text { color: #00FFBB; }
        .sub-hero { font-size: 1.3rem !important; color: #A0AEC0; margin-bottom: 2rem; }
        .price-card { background-color: #1A202C; padding: 2.2rem; border-radius: 10px; border: 1px solid #2D3748; height: 100%; }
        .price-card-premium { background-color: #1A202C; padding: 2.2rem; border-radius: 10px; border: 2px solid #00FFBB; height: 100%; }
    </style>
""", unsafe_allow_html=True)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

JURISDICTION_PROFILES = {
    "UK":     {"csv_key": "UK",     "data_dir": "uk", "generic_label": "VMP Name",            "brand_label": "AMP Name",          "id_label": "VMP/AMP ID"},
    "US":     {"csv_key": "US",     "data_dir": "us", "generic_label": "Established Name",    "brand_label": "Proprietary Name",  "id_label": "NDC Number"},
    "Canada": {"csv_key": "CANADA", "data_dir": "ca", "generic_label": "Active Ingredient",   "brand_label": "Brand Name",        "id_label": "DIN"},
}

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

DATA_VERSION = "v8"

_PHARMA_OVERRIDE_TARGETS: frozenset[str] = frozenset({
    "paracetamol", "acetaminophen", "ibuprofen", "metformin", "gabapentin",
    "amoxicillin", "tramadol", "levothyroxine", "amlodipine", "lisinopril",
    "atorvastatin", "simvastatin", "omeprazole", "losartan", "pantoprazole",
    "sertraline", "fluoxetine", "prednisone", "metoprolol", "albuterol",
    "amoxicillin/clavulanate", "prednisone",
})

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
        pass

def clear_cache_db():
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass
    if "ai_cache" in st.session_state:
        st.session_state["ai_cache"] = {}

def _cache_is_plausible(clean_token: str, generic_name: str) -> bool:
    c = re.sub(r"[^a-z]", "", clean_token.lower())
    g = re.sub(r"[^a-z]", "", str(generic_name).lower())
    if not c or not g:
        return True
    if len(c) <= 4:
        return g in _PHARMA_OVERRIDE_TARGETS
    prefix = c[: min(5, len(c))]
    return g.startswith(prefix) or c.startswith(g[: min(5, len(g))])

# =====================================================================
# 3. NOISE STRIPPING
# =====================================================================
_STRIP_NOISE = re.compile(
    r'\s*//\s*batch[-\s]?\w+'
    r'|\b\d+(?:\.\d+)?(?:/\d+)?\s*(?:mg|mcg|g|ml|iu)\b'
    r'|\b(?:tab|tabs|cap|caps|er|sr|xr|xl|dr|eff|inh|inj|soln?|susp|tar|hcl|hci)\b'
    r'|\bx-\d+\b',
    re.IGNORECASE
)

def strip_noise(raw: str) -> str:
    name = _STRIP_NOISE.sub(" ", raw).strip()
    name = re.sub(r'\b[a-zA-Z]\b', '', name).strip()
    return " ".join(name.split())

# =====================================================================
# 4. CONTEXT-AWARE DOSAGE MATCHING (NEW LAYER 1 ENHANCEMENT)
# =====================================================================
_DOSAGE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml|iu|units?)\b',
    re.IGNORECASE
)
_FORM_PATTERN = re.compile(
    r'\b(tab|tabs|cap|caps|capsule|capsules|'
    r'er|sr|xr|xl|dr|eff|effervescent|inj|injection|'
    r'soln?|solution|susp|suspension|cream|gel|ointment|patch|spray)\b',
    re.IGNORECASE
)

def extract_dosage_context(raw_text: str) -> dict:
    """Extract dosage strength and form from invoice text."""
    context = {"strength": None, "form": None, "raw": raw_text}
    
    strength_match = _DOSAGE_PATTERN.search(raw_text)
    if strength_match:
        context["strength"] = strength_match.group(0).lower()
    
    form_match = _FORM_PATTERN.search(raw_text)
    if form_match:
        context["form"] = form_match.group(1).lower()
    
    return context

def score_registry_match_with_context(
    resolved_name: str,
    context: dict,
    master_df: pd.DataFrame
) -> tuple[str, float]:
    """Score registry matches based on dosage/form alignment."""
    if master_df.empty or not resolved_name:
        return resolved_name, 0.0
    
    matching_rows = master_df[
        master_df["generic_name"].astype(str).str.lower().str.contains(
            resolved_name.lower(), regex=False, na=False
        )
    ]
    
    if matching_rows.empty:
        return resolved_name, 0.0
    
    if len(matching_rows) == 1:
        return str(matching_rows.iloc[0]["generic_name"]), 0.0
    
    best_match = resolved_name
    best_score = 0.0
    
    for _, row in matching_rows.iterrows():
        registry_name = str(row["generic_name"]).lower()
        score = 0.0
        
        if context.get("strength") and context["strength"] in registry_name:
            score += 0.5
        
        if context.get("form") and context["form"] in registry_name:
            score += 0.3
        
        if score > best_score or (score == best_score and len(registry_name) < len(best_match.lower())):
            best_score = score
            best_match = str(row["generic_name"])
    
    return best_match, best_score

# =====================================================================
# 5. MASTER REGISTRY LOADER
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

_NA_STRINGS = re.compile(r'^(nan|none|n/a|na|-)$', re.IGNORECASE)

def _normalize_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip().str.upper()
    for col in ["generic_name", "brand_name", "baseline_price", "system_id"]:
        if col not in df.columns:
            df[col] = "N/A"
    for col in ("brand_name", "system_id"):
        df[col] = (
            df[col].astype(str).str.strip()
            .apply(lambda v: "N/A" if _NA_STRINGS.match(v) else v)
        )
    return df

def _base_ingredient(name: str) -> str:
    text = strip_noise(str(name))
    text = _SALT_SUFFIX.sub("", text).strip()
    text = re.split(r"[/+]", text)[0].strip()
    return " ".join(w.capitalize() for w in text.split())

@st.cache_data
def _load_full_registry(_version: str = DATA_VERSION) -> pd.DataFrame:
    csv_path = os.path.join(BASE_PATH, "master_registry.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return _normalize_registry_columns(pd.read_csv(csv_path))
    except Exception:
        return pd.DataFrame()

@st.cache_data
def _load_pharmacy_core(region_key: str, data_dir: str, _version: str = DATA_VERSION) -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "data", f"pharmacy_core_{data_dir}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = _normalize_registry_columns(pd.read_csv(path))
        df["region"] = region_key
        return df.drop_duplicates(subset=["generic_name"], keep="first")
    except Exception:
        return pd.DataFrame()

def _merge_registry_sources(
    base_reg: pd.DataFrame, core: pd.DataFrame, supplement: pd.DataFrame
) -> pd.DataFrame:
    tagged = []
    for label, df in (("base", base_reg), ("core", core), ("regional", supplement)):
        if df is None or df.empty:
            continue
        part = df.copy()
        part["_src"] = label
        tagged.append(part)
    if not tagged:
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id"])

    pool = pd.concat(tagged, ignore_index=True)
    pool["generic_key"] = pool["generic_name"].astype(str).str.lower().str.strip()
    rows = []
    for _, group in pool.groupby("generic_key", sort=False):
        group = group.copy()
        row = group.iloc[0].copy()
        priced = group[(group["_src"] == "base") & (pd.to_numeric(group["baseline_price"], errors="coerce").fillna(0) > 0)]
        if not priced.empty:
            row["baseline_price"] = priced.iloc[0]["baseline_price"]
        for src in ("regional", "core", "base"):
            branded = group[
                (group["_src"] == src)
                & (~group["brand_name"].astype(str).str.upper().isin(["N/A", "NAN", "NONE", ""]))
            ]
            if not branded.empty:
                row["brand_name"] = branded.iloc[0]["brand_name"]
                row["system_id"] = branded.iloc[0]["system_id"]
                break
        rows.append(row)
    merged = pd.DataFrame(rows).drop(columns=["_src", "generic_key"], errors="ignore")
    return merged.drop_duplicates(subset=["generic_name"], keep="first").reset_index(drop=True)

@st.cache_data
def _load_regional_supplement(region_key: str, data_dir: str, _version: str = DATA_VERSION) -> pd.DataFrame:
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

    core       = _load_pharmacy_core(target_key, data_dir)
    supplement = _load_regional_supplement(target_key, data_dir)
    merged     = _merge_registry_sources(base_reg, core, supplement)

    if merged.empty:
        st.sidebar.error("No registry data found for this region.")
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id"])

    has_regional = not supplement.empty
    has_core     = not core.empty
    st.sidebar.success(
        f"Loaded {len(merged):,} drugs for {region} "
        f"({len(base_reg):,} priced · {len(core):,} core · "
        f"{'FDA/NHS linked' if has_regional else 'core only'})"
    )
    return merged

@st.cache_data
def build_registry_index(_registry_key: str, generic_names: tuple, _version: str = DATA_VERSION) -> dict[str, int]:
    index: dict[str, int] = {}
    synonyms = REGIONAL_SYNONYMS.get(_registry_key, {})

    override_lower = {v.lower(): k for k, v in _PHARMA_OVERRIDES.items()}
    for i, generic in enumerate(generic_names):
        g_lower = str(generic).lower().strip()
        if g_lower in override_lower:
            abbrev_key = override_lower[g_lower]
            if abbrev_key not in index:
                index[abbrev_key] = i

    for i, generic in enumerate(generic_names):
        generic = str(generic)
        keys = {generic.lower(), _base_ingredient(generic).lower()}
        for key in keys:
            if not key:
                continue
            if len(key) <= 4:
                continue
            if key not in index or len(generic) < len(str(generic_names[index[key]])):
                index[key] = i
            alt = synonyms.get(key)
            if alt and (alt not in index or len(generic) < len(str(generic_names[index[alt]]))):
                index[alt] = i
        first = generic.lower().split()[0] if generic.split() else ""
        if len(first) >= 6 and (first not in index or len(generic) < len(str(generic_names[index[first]]))):
            index[first] = i
    return index

def resolve_in_registry(
    candidate: str,
    master_df: pd.DataFrame,
    registry_index: dict[str, int],
    region_key: str,
    *,
    from_abbrev: bool = False,
) -> tuple[str | None, int | None]:
    if not candidate or master_df.empty:
        return None, None

    c_lower = candidate.lower().strip()

    if not from_abbrev and c_lower in _PHARMA_OVERRIDES:
        return None, None

    synonyms = REGIONAL_SYNONYMS.get(region_key, {})
    probes = [
        c_lower,
        _base_ingredient(candidate).lower(),
        synonyms.get(c_lower, ""),
    ]
    first_word = c_lower.split()[0] if c_lower.split() else ""
    if first_word and (from_abbrev or len(first_word) >= 6):
        probes.append(first_word)

    for probe in probes:
        if probe and probe in registry_index:
            idx = registry_index[probe]
            resolved = str(master_df.iloc[idx]["generic_name"])
            if len(probe) <= 5 and not from_abbrev:
                if len(resolved) > 40 or any(c in resolved for c in "()-,/[]"):
                    continue
            return resolved, idx

    best_idx, best_len = None, 0
    for probe in probes:
        if not probe or len(probe) < 6:
            continue
        for key, idx in registry_index.items():
            if key.startswith(probe) and len(key) > best_len:
                best_idx, best_len = idx, len(key)
    if best_idx is not None:
        return str(master_df.iloc[best_idx]["generic_name"]), best_idx
    return None, None

# =====================================================================
# 6. LAYER 1: DETERMINISTIC ABBREVIATION LOOKUP
# =====================================================================
@st.cache_data
def _load_abbrev_dict(_version: str = DATA_VERSION) -> dict:
    try:
        with open(ABBREV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

_PHARMA_OVERRIDES = {
    "pcm":    "Paracetamol",
    "ibu":    "Ibuprofen",
    "met":    "Metformin",
    "gab":    "Gabapentin",
    "gaba":   "Gabapentin",
    "para":   "Paracetamol",
    "amox":   "Amoxicillin",
    "tram":   "Tramadol",
    "levo":   "Levothyroxine",
    "amlo":   "Amlodipine",
    "lisino": "Lisinopril",
    "atorva": "Atorvastatin",
    "simva":  "Simvastatin",
    "omepra": "Omeprazole",
    "losart": "Losartan",
    "panto":  "Pantoprazole",
    "sertra": "Sertraline",
    "fluox":  "Fluoxetine",
    "predni": "Prednisone",
    "metop":  "Metoprolol",
    "metfor": "Metformin",
    "albu":   "Albuterol",
    "salbu":  "Albuterol",
    "levoth": "Levothyroxine",
    "sertr":  "Sertraline",
    "fluoxe": "Fluoxetine",
    "prednis":"Prednisone",
    "metopro":"Metoprolol",
}

@st.cache_data
def _build_abbrev_lookup(registry_key: str, registry_names: tuple, _version: str = DATA_VERSION) -> tuple:
    combined: dict[str, str] = dict(_PHARMA_OVERRIDES)

    for k, v in _load_abbrev_dict().items():
        combined.setdefault(k, v)

    _SKIP_CHARS = re.compile(r'[,()/\[\]+]')
    for name in registry_names:
        clean = str(name).lower().strip()
        if len(clean) < 7 or len(clean) > 40 or _SKIP_CHARS.search(clean):
            continue
        for n in (5, 6):
            combined.setdefault(clean[:n], str(name))

    sorted_items = tuple(sorted(combined.items(), key=lambda x: len(x[0]), reverse=True))
    return sorted_items

def abbrev_lookup(clean_token: str, abbrev_items: tuple) -> str | None:
    t = clean_token.lower().strip()
    if len(t) <= 4:
        for abbrev, full_name in abbrev_items:
            if abbrev.lower().strip() == t:
                return full_name
        return None
    for abbrev, full_name in abbrev_items:
        if re.match(r'^' + re.escape(abbrev.lower().strip()) + r'(\s|/|$)', t, re.IGNORECASE):
            return full_name
    return None

def abbrev_lookup_fallback(clean_token: str, abbrev_items: tuple, master_df: pd.DataFrame,
                          context: dict | None = None) -> tuple[str | None, str | None]:
    abbrev_result = abbrev_lookup(clean_token, abbrev_items)
    if not abbrev_result:
        return None, None

    if abbrev_result.lower() in _PHARMA_OVERRIDE_TARGETS:
        if context:
            refined_name, _ = score_registry_match_with_context(abbrev_result, context, master_df)
            return refined_name, None
        return abbrev_result, None

    if master_df.empty:
        return None, None

    registry_names = master_df["generic_name"].astype(str).str.lower().str.strip().tolist()
    if abbrev_result.lower() in registry_names:
        if context:
            refined_name, _ = score_registry_match_with_context(abbrev_result, context, master_df)
            return refined_name, None
        return abbrev_result, None

    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(registry_names)
        query = vectorizer.transform([abbrev_result.lower()])
        scores = cosine_similarity(query, matrix)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= 0.55:
            matched_name = master_df.iloc[best_idx]["generic_name"]
            if context:
                refined_name, _ = score_registry_match_with_context(matched_name, context, master_df)
                return refined_name, f"abbrev→{abbrev_result}"
            return matched_name, f"abbrev→{abbrev_result}"
    except Exception:
        pass

    return None, None

# =====================================================================
# 7. LAYER 2: TF-IDF COSINE SIMILARITY
# =====================================================================
@st.cache_resource
def build_tfidf_index(registry_key: str, registry_names: tuple, _version: str = DATA_VERSION):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    matrix     = vectorizer.fit_transform([n.lower() for n in registry_names])
    return vectorizer, matrix

def tfidf_match_batch(
    clean_tokens: list[str],
    master_df: pd.DataFrame,
    region_key: str,
    threshold: float,
    contexts: list[dict] | None = None,
) -> list[tuple[str | None, float]]:
    n = len(clean_tokens)
    if master_df.empty or n == 0:
        return [(None, 0.0)] * n

    generic_names_tuple = tuple(master_df["generic_name"].astype(str).tolist())

    try:
        vectorizer, matrix = build_tfidf_index(region_key, generic_names_tuple, DATA_VERSION)
        results: list[tuple[str | None, float]] = []
        chunk_size = 2000
        for start in range(0, n, chunk_size):
            queries = [t.lower() for t in clean_tokens[start:start + chunk_size]]
            score_matrix = cosine_similarity(vectorizer.transform(queries), matrix)
            for idx, row_scores in enumerate(score_matrix):
                best_idx   = int(np.argmax(row_scores))
                best_score = float(row_scores[best_idx])
                if best_score >= threshold:
                    matched_name = str(master_df.iloc[best_idx]["generic_name"])
                    if contexts and (start + idx) < len(contexts) and contexts[start + idx]:
                        refined_name, _ = score_registry_match_with_context(
                            matched_name, contexts[start + idx], master_df
                        )
                        results.append((refined_name, best_score))
                    else:
                        results.append((matched_name, best_score))
                else:
                    results.append((None, best_score))
        return results
    except Exception:
        return [(None, 0.0)] * n

# =====================================================================
# 8. LAYER 3: GEMINI AI FALLBACK
# =====================================================================
def call_gemini_api(raw_input: str, clean_token: str,
                    registry_names: list, jurisdiction: str) -> dict:
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
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result_text = response.choices[0].message.content.strip()
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"status": "error", "reason": "Invalid JSON response from API"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}

# =====================================================================
# 9. MAIN ROUTING LOGIC - LANDING PAGE, LOGIN, AND APP ENGINE
# =====================================================================

# LANDING PAGE (Unauthenticated Users)
if st.session_state.page == "landing":
    
    # Navigation Bar
    nav_col1, nav_col2 = st.columns([8, 2])
    with nav_col1:
        st.markdown("### 🧬 **APEX LOGIC**")
    with nav_col2:
        if st.button("🔒 Client Login Portal", use_container_width=True, type="secondary"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("---")

    # Hero Section
    hero_col1, hero_col2 = st.columns([6, 4])
    with hero_col1:
        st.markdown('<p class="hero-title">Stop Manually Wrestling with <span class="accent-text">Pharmacy Vendor Invoices</span>.</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-hero">The automated data harmonization engine built for modern pharmacies. Instantly scrub, standardize, and format messy supplier spreadsheets across the UK, Canada, and the US.</p>', unsafe_allow_html=True)
    with hero_col2:
        st.info(
            "📦 **Data Pipeline Ingestion Stream**\n\n"
            "→ `PCM 500mg (UK)` *(Messy Vendor Invoiced Shorthand)*\n\n"
            "⚡ *Apex Harmonization Matrix Processing...*\n\n"
            "→ **Paracetamol 500mg Tablet (Active Canonical ID: 104)**"
        )

    st.markdown("##")
    st.markdown("---")
    
    # Value Proposition
    st.markdown("### 🛠️ Enterprise Data Infrastructure")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("#### 🔍 Who We Are")
        st.write("An advanced data-cleaning and harmonization layer engineered specifically to eliminate structural friction in complex pharmaceutical supply chains.")
    with p2:
        st.markdown("#### ⚡ What It Does")
        st.write("We ingest completely unstandardized, abbreviated retail pharmacy invoices. The platform instantly purges human errors, shorthand typos, and mismatched naming conventions.")
    with p3:
        st.markdown("#### 🚀 Why We Are Better")
        st.write("Zero manual labor. Our framework scales effortlessly across Health Canada, UK NHS, and US baseline registries with an evolutionary dynamic caching layer.")

    st.markdown("##")
    st.markdown("---")

    # Scarcity Banner
    st.warning(
        "🚀 **Global Pilot Group Initiative (Strictly Limited)**: To seed our foundational pilot group across our launch regions, "
        "We are offering an exclusive **50% lifetime discount** to the first **5 pharmacies** to register in the UK, Canada, and the US. "
        "Use code **`APEXEARLY50`** at checkout. Once the 5th slot in your region is filled, the system automatically reverts to regular pricing."
    )

    # Pricing Tiers
    st.markdown("### 💳 Corporate Deployment Packages")
    tier1_col, tier2_col, tier3_col = st.columns(3)
    
    with tier1_col:
        st.markdown("""
        <div class="price-card">
            <h4>📦 Tier 1: Standalone Toolkit</h4>
            <h2>£99 <span style="font-size:1rem;color:#A0AEC0">One-Time Payment</span></h2>
            <hr style="border-color:#2D3748">
            <p style="margin-bottom:0.5rem;">• Full master abbreviation mapping package (abbreviations.json)</p>
            <p style="margin-bottom:0.5rem;">• Local pipeline implementation and audit templates</p>
            <p style="margin-bottom:0.5rem;">• Optimized for internal development and localized audit setups</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Get Standalone Kit", "https://whop.com/your-tier-1-checkout-link", use_container_width=True)

    with tier2_col:
        st.markdown("""
        <div class="price-card-premium">
            <h4>🧬 Tier 2: Cloud Instance Software Access</h4>
            <h2>£499 <span style="font-size:1rem;color:#A0AEC0">/ Month</span></h2>
            <hr style="border-color:#00FFBB">
            <p style="margin-bottom:0.5rem;">• Infinite, automated 24/7 web application invoice uploads</p>
            <p style="margin-bottom:0.5rem;">• Multi-regional database compliance (UK, Canada, US formats)</p>
            <p style="margin-bottom:0.5rem;">• Absolute cross-border matrix validation engine</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Launch Cloud Engine", "https://whop.com/your-tier-2-checkout-link", type="primary", use_container_width=True)

    with tier3_col:
        st.markdown("""
        <div class="price-card">
            <h4>🏢 Tier 3: Bespoke Enterprise Integration</h4>
            <h2>£2,500 <span style="font-size:1rem;color:#A0AEC0">Setup</span> + £250<span style="font-size:1rem;color:#A0AEC0">/Mo</span></h2>
            <hr style="border-color:#2D3748">
            <p style="margin-bottom:0.5rem;">• Custom logic mapped directly to your specific PMS layouts</p>
            <p style="margin-bottom:0.5rem;">• Secure automated webhooks and pipeline scripts</p>
            <p style="margin-bottom:0.5rem;">• Priority server allocation & dedicated architecture kickoff strategy call</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Book Integration Call", "https://calendly.com/your-link", use_container_width=True)

# LOGIN PAGE (Authentication Gateway)
elif st.session_state.page == "login" and not st.session_state.authenticated:
    st.markdown("### 🔒 Enterprise System Authentication")
    st.write("Please authenticate your active subscription to unlock the multi-regional cleaning infrastructure.")
    
    input_key = st.text_input("Enter your Whop Membership Key:", type="password")
    
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        if st.button("Unlock Core Dashboard", type="primary", use_container_width=True):
            if check_whop_authorization(input_key):
                st.session_state.authenticated = True
                st.session_state.page = "app"
                st.success("Access Authorized. Initializing engine matrix...")
                st.rerun()
            else:
                st.error("Access Denied. Invalid, expired, or inactive membership key.")
    with l_col2:
        if st.button("⬅️ Back to Storefront Overview", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

# PROTECTED APP ENGINE (Authenticated Users)
if st.session_state.authenticated or st.session_state.page == "app":
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🧬 Session Controls")
        if st.button("🚪 Secure Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.page = "landing"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        selected_region = st.selectbox("Select Jurisdiction", list(JURISDICTION_PROFILES.keys()))
        
        st.markdown("---")
        st.markdown("### 📊 Cache Management")
        if st.button("🔄 Clear Cache Database", use_container_width=True):
            clear_cache_db()
            st.success("Cache cleared successfully!")
            st.rerun()
    
    st.title("▲ Apex Logic Software Core")
    st.success("🔒 Secure Pipeline Verification Active (UK / CA / US Master Matrices Engaged)")
    
    # Load Registry
    master_df = load_jurisdictional_registry(selected_region)
    
    if master_df.empty:
        st.error("Unable to load registry data. Please check your configuration.")
    else:
        # Main Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📤 Invoice Upload", "🔍 Manual Lookup", "📊 Batch Processing", "📈 Analytics"])
        
        with tab1:
            st.markdown("### 📤 Single Invoice Harmonization")
            st.write("Upload or paste a single invoice entry for real-time harmonization.")
            
            input_method = st.radio("Input Method", ["Paste Text", "Upload CSV"])
            
            if input_method == "Paste Text":
                raw_input = st.text_area("Paste invoice entry:", height=100)
                
                if st.button("Harmonize Entry", type="primary", use_container_width=True):
                    if raw_input.strip():
                        with st.spinner("Processing..."):
                            clean_token = strip_noise(raw_input)
                            context = extract_dosage_context(raw_input)
                            
                            generic_names_tuple = tuple(master_df["generic_name"].astype(str).tolist())
                            abbrev_items = _build_abbrev_lookup(selected_region, generic_names_tuple)
                            registry_index = build_registry_index(selected_region, generic_names_tuple)
                            
                            # Layer 1: Abbreviation Lookup
                            resolved, _ = abbrev_lookup_fallback(clean_token, abbrev_items, master_df, context)
                            
                            if not resolved:
                                # Layer 2: TF-IDF
                                results = tfidf_match_batch([clean_token], master_df, selected_region, 0.65, [context])
                                if results and results[0][0]:
                                    resolved = results[0][0]
                            
                            if resolved:
                                st.success(f"✅ Resolved: **{resolved}**")
                                if context.get("strength"):
                                    st.info(f"📋 Detected Strength: {context['strength']}")
                                if context.get("form"):
                                    st.info(f"💊 Detected Form: {context['form']}")
                            else:
                                st.warning("⚠️ No match found in registry. Consider Layer 3 AI fallback.")
            
            else:
                uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    if st.button("Process Batch", type="primary", use_container_width=True):
                        st.info("Batch processing initiated...")
        
        with tab2:
            st.markdown("### 🔍 Manual Drug Lookup")
            search_term = st.text_input("Search for drug name:")
            
            if search_term:
                matches = master_df[
                    master_df["generic_name"].astype(str).str.lower().str.contains(
                        search_term.lower(), regex=False, na=False
                    )
                ].head(10)
                
                if not matches.empty:
                    st.dataframe(matches[["generic_name", "brand_name", "baseline_price", "system_id"]])
                else:
                    st.warning("No matches found.")
        
        with tab3:
            st.markdown("### 📊 Batch Processing Engine")
            st.write("Process multiple invoices at once for enterprise-scale harmonization.")
            
            batch_input = st.text_area("Paste multiple entries (one per line):", height=200)
            
            if st.button("Process Batch", type="primary", use_container_width=True):
                if batch_input.strip():
                    entries = [e.strip() for e in batch_input.split("\n") if e.strip()]
                    
                    with st.spinner(f"Processing {len(entries)} entries..."):
                        clean_tokens = [strip_noise(e) for e in entries]
                        contexts = [extract_dosage_context(e) for e in entries]
                        
                        generic_names_tuple = tuple(master_df["generic_name"].astype(str).tolist())
                        results = tfidf_match_batch(clean_tokens, master_df, selected_region, 0.65, contexts)
                        
                        output_data = []
                        for i, (entry, result) in enumerate(zip(entries, results)):
                            output_data.append({
                                "Original": entry,
                                "Cleaned": clean_tokens[i],
                                "Resolved": result[0] if result[0] else "NO MATCH",
                                "Confidence": f"{result[1]:.2%}" if result[1] else "0%"
                            })
                        
                        output_df = pd.DataFrame(output_data)
                        st.dataframe(output_df, use_container_width=True)
                        
                        csv = output_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="harmonization_results.csv",
                            mime="text/csv"
                        )
        
        with tab4:
            st.markdown("### 📈 System Analytics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Drugs in Registry", f"{len(master_df):,}")
            with col2:
                st.metric("Region", selected_region)
            with col3:
                st.metric("Data Version", DATA_VERSION)
            
            st.markdown("---")
            st.markdown("#### 📊 Registry Composition")
            
            if "region" in master_df.columns:
                region_counts = master_df["region"].value_counts()
                st.bar_chart(region_counts)
            
            st.markdown("#### 🏷️ Brand Name Coverage")
            branded = master_df[master_df["brand_name"] != "N/A"]
            st.metric("Drugs with Brand Names", f"{len(branded):,} ({len(branded)/len(master_df)*100:.1f}%)")
            
            st.markdown("#### 💰 Pricing Data")
            priced = master_df[pd.to_numeric(master_df["baseline_price"], errors="coerce") > 0]
            st.metric("Drugs with Pricing", f"{len(priced):,} ({len(priced)/len(master_df)*100:.1f}%)")
