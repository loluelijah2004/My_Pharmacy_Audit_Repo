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
# 0. PLATFORM CONFIGURATION & JURISDICTIONAL TAXONOMY
# =====================================================================
st.set_page_config(
    page_title="Apex Logic | Professional Invoice Harmonization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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

# DATA_VERSION — bump this string whenever master_registry.csv or
# abbreviations.json changes. This busts all @st.cache_data caches
# that depend on registry/abbreviation data so Streamlit picks up
# the new file contents immediately without a server restart.
DATA_VERSION = "v7"

# Canonical targets for _PHARMA_OVERRIDES — defined early so _cache_is_plausible
# can reference them without a forward-reference NameError.
# Keep in sync with _PHARMA_OVERRIDES below.
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
        pass  # Streamlit Cloud read-only FS — fails silently, session cache still works


def clear_cache_db():
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass
    if "ai_cache" in st.session_state:
        st.session_state["ai_cache"] = {}


def _cache_is_plausible(clean_token: str, generic_name: str) -> bool:
    """Reject stale cache rows that mapped invoice text to the wrong molecule.

    BUG FIX: Previously returned False unconditionally for tokens ≤4 chars,
    which meant correct cached results for 'PARA', 'MET', 'IBU' etc. were
    always bypassed, forcing expensive re-resolution every run.
    Now: short tokens are only rejected if the cached name is clearly wrong
    (i.e. the cached generic name does not appear in _PHARMA_OVERRIDES values).
    """
    c = re.sub(r"[^a-z]", "", clean_token.lower())
    g = re.sub(r"[^a-z]", "", str(generic_name).lower())
    if not c or not g:
        return True
    if len(c) <= 4:
        # Accept cache hit if the cached name is a known pharma override target
        return g in _PHARMA_OVERRIDE_TARGETS
    prefix = c[: min(5, len(c))]
    return g.startswith(prefix) or c.startswith(g[: min(5, len(g))])

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


# BUG FIX (Brand names always N/A): pandas read_csv converts empty cells to
# float NaN which becomes the string "nan" after .astype(str). The old code
# used dict .replace() which only matches exact whole-cell values. Using
# regex=False str.replace is safer; we also normalise "none" and whitespace.
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
    """Reduce a registry label to a matchable ingredient token."""
    text = strip_noise(str(name))
    text = _SALT_SUFFIX.sub("", text).strip()
    text = re.split(r"[/+]", text)[0].strip()
    return " ".join(w.capitalize() for w in text.split())

@st.cache_data
def _load_full_registry(_version: str = DATA_VERSION) -> pd.DataFrame:
    """Load master_registry.csv. _version param busts cache when DATA_VERSION changes."""
    csv_path = os.path.join(BASE_PATH, "master_registry.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return _normalize_registry_columns(pd.read_csv(csv_path))
    except Exception:
        return pd.DataFrame()

@st.cache_data
def _load_pharmacy_core(region_key: str, data_dir: str, _version: str = DATA_VERSION) -> pd.DataFrame:
    """Compact bundled list of common pharmacy drugs (brand + NDC) for cloud deploys."""
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
    """Merge priced baseline, core pharmacy list, and regional FDA/NHS data."""
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
        # Priced baseline from master_registry takes priority
        priced = group[(group["_src"] == "base") & (pd.to_numeric(group["baseline_price"], errors="coerce").fillna(0) > 0)]
        if not priced.empty:
            row["baseline_price"] = priced.iloc[0]["baseline_price"]
        # Brand + NDC from core/regional (master_registry has no proprietary names)
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

    core       = _load_pharmacy_core(target_key, data_dir)
    supplement = _load_regional_supplement(target_key, data_dir)
    merged     = _merge_registry_sources(base_reg, core, supplement)

    if merged.empty:
        st.sidebar.error("❌ No registry data found for this region.")
        return pd.DataFrame(columns=["generic_name", "brand_name", "baseline_price", "system_id"])

    has_regional = not supplement.empty
    has_core     = not core.empty
    st.sidebar.success(
        f"✅ {len(merged):,} drugs loaded for {region} "
        f"({len(base_reg):,} priced · {len(core):,} core · "
        f"{'FDA/NHS linked' if has_regional else 'core only — upload data/{data_dir}/ for full NDC'})"
    )
    if not has_core and not has_regional:
        st.sidebar.warning(
            "⚠️ Only baseline prices loaded — common drug names and proprietary labels may be missing."
        )
    return merged

@st.cache_data
def build_registry_index(_registry_key: str, generic_names: tuple, _version: str = DATA_VERSION) -> dict[str, int]:
    """Maps normalized ingredient tokens to row indices for O(1) Layer 1 resolution.

    BUG FIX: Previously, auto-prefix indexing ran over ALL registry names
    including obscure chemicals like 'Para-Iodo-D-Phenylalanine Hydroxamic Acid',
    which generated a 'para' key that overwrote the _PHARMA_OVERRIDES entry.
    Fixes applied:
      1. Short tokens (≤4 chars) are NEVER added to the auto-prefix index.
      2. _PHARMA_OVERRIDES are seeded into the index FIRST so they cannot be
         overwritten by any auto-generated prefix from an obscure registry name.
      3. First-token index threshold raised to 6 chars (was 5) for extra safety.
    """
    index: dict[str, int] = {}
    synonyms = REGIONAL_SYNONYMS.get(_registry_key, {})

    # ── Seed _PHARMA_OVERRIDES first so obscure registry names cannot overwrite them ──
    # We map each override target name to the first registry row that matches it.
    override_lower = {v.lower(): k for k, v in _PHARMA_OVERRIDES.items()}  # e.g. "paracetamol" → "para"
    for i, generic in enumerate(generic_names):
        g_lower = str(generic).lower().strip()
        if g_lower in override_lower:
            # Pin the abbreviation key → this registry row (only if not already pinned)
            abbrev_key = override_lower[g_lower]
            if abbrev_key not in index:
                index[abbrev_key] = i

    # ── Main index build ──
    for i, generic in enumerate(generic_names):
        generic = str(generic)
        keys = {generic.lower(), _base_ingredient(generic).lower()}
        for key in keys:
            if not key:
                continue
            # BUG FIX: never let a short key (≤4 chars) be auto-indexed from a
            # long obscure name — short keys must come from _PHARMA_OVERRIDES only.
            if len(key) <= 4:
                continue
            if key not in index or len(generic) < len(str(generic_names[index[key]])):
                index[key] = i
            alt = synonyms.get(key)
            if alt and (alt not in index or len(generic) < len(str(generic_names[index[alt]]))):
                index[alt] = i
        # First-token index only for longer tokens (≥6 chars prevents PARA/MET/GABA collisions)
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
    """Map an abbreviation or cleaned token to the best registry row.

    CRITICAL FIX: Short tokens that are known pharma abbreviations (≤6 chars,
    in _PHARMA_OVERRIDES keys) must NOT be resolved via the registry index
    directly — they must come through abbrev_lookup_fallback first.
    This prevents 'PARA' hitting the index and matching 'Para-Iodo-D-...'
    even after the index is rebuilt correctly, as a defence-in-depth guard.

    When from_abbrev=True, the candidate is already the resolved full name
    (e.g. 'Paracetamol'), so index lookup is appropriate.
    """
    if not candidate or master_df.empty:
        return None, None

    c_lower = candidate.lower().strip()

    # GUARD: if the raw token is a known pharma abbreviation key and we're NOT
    # coming from the abbreviation resolver, block direct index lookup.
    # This prevents 'PARA' → 'Para-Iodo...' via Exact Name Match.
    if not from_abbrev and c_lower in _PHARMA_OVERRIDES:
        return None, None

    synonyms = REGIONAL_SYNONYMS.get(region_key, {})
    probes = [
        c_lower,
        _base_ingredient(candidate).lower(),
        synonyms.get(c_lower, ""),
    ]
    first_word = c_lower.split()[0] if c_lower.split() else ""
    # Short invoice tokens (PARA, MET) must go through abbreviation map, not token index.
    # Only add first_word probe for longer tokens (≥6 chars) or when coming from abbrev resolver.
    if first_word and (from_abbrev or len(first_word) >= 6):
        probes.append(first_word)

    for probe in probes:
        if probe and probe in registry_index:
            idx = registry_index[probe]
            resolved = str(master_df.iloc[idx]["generic_name"])
            # Extra safety: if the resolved name is an obscure chemical (very long or
            # contains special chars) and the probe is short, reject it.
            if len(probe) <= 5 and not from_abbrev:
                if len(resolved) > 40 or any(c in resolved for c in "()-,/[]"):
                    continue
            return resolved, idx

    # Prefix: registry name extends the probe (e.g. "lisinopril" → "lisinopril hydrochloride")
    best_idx, best_len = None, 0
    for probe in probes:
        if not probe or len(probe) < 6:  # Raised from 4 to 6 — short probes cause wrong prefix hits
            continue
        for key, idx in registry_index.items():
            if key.startswith(probe) and len(key) > best_len:
                best_idx, best_len = idx, len(key)
    if best_idx is not None:
        return str(master_df.iloc[best_idx]["generic_name"]), best_idx
    return None, None

# =====================================================================
# 6. LAYER 1: DETERMINISTIC ABBREVIATION LOOKUP
#    Loads custom abbreviations from an external abbreviations.json file.
# =====================================================================

@st.cache_data
def _load_abbrev_dict(_version: str = DATA_VERSION) -> dict:
    """Load abbreviations.json. _version param busts cache when DATA_VERSION changes."""
    try:
        with open(ABBREV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Wholesaler shorthands that collide with obscure registry ingredient names.
# BUG FIX: Added missing shorthands seen in enterprise_procurement.csv:
#   LISINO → Lisinopril, LEVO → Levothyroxine, SIMVA → Simvastatin,
#   AMLO → Amlodipine, GABA → Gabapentin, ATORVA → Atorvastatin
# These are seeded into build_registry_index BEFORE auto-prefix generation
# so obscure chemical names cannot overwrite them.
_PHARMA_OVERRIDES = {
    # 3-4 char codes (highest collision risk — must be in overrides only)
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
    # 5-6 char codes
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
    # Extra common shorthands
    "metfor": "Metformin",
    "albu":   "Albuterol",
    "salbu":  "Albuterol",
    "losart": "Losartan",
    "levoth": "Levothyroxine",
    "sertr":  "Sertraline",
    "fluoxe": "Fluoxetine",
    "prednis":"Prednisone",
    "metopro":"Metoprolol",
}


@st.cache_data
def _build_abbrev_lookup(registry_key: str, registry_names: tuple, _version: str = DATA_VERSION) -> tuple:
    """Merge JSON abbreviations with auto-generated registry prefixes.

    BUG FIX: Auto-prefix generation previously created entries like
    'para' → 'Para-Iodo-D-Phenylalanine Hydroxamic Acid' from the registry,
    which then overwrote the correct _PHARMA_OVERRIDES entry via setdefault.
    Fixes:
      1. _PHARMA_OVERRIDES are applied FIRST (not last via .update()).
         setdefault() then prevents any auto-prefix from overwriting them.
      2. Auto-prefix generation is skipped for names containing special chars
         (, / + [ ) or names shorter than 7 chars to avoid short collisions.
      3. Only 5 and 6-char prefixes are auto-generated (4-char is too risky).
    """
    # Seed with _PHARMA_OVERRIDES first — these are authoritative and must not
    # be overwritten by anything from the registry or JSON file.
    combined: dict[str, str] = dict(_PHARMA_OVERRIDES)

    # Layer in JSON abbreviations (setdefault = don't overwrite pharma overrides)
    for k, v in _load_abbrev_dict().items():
        combined.setdefault(k, v)

    # Auto-generate 5 and 6-char prefixes from clean registry names only
    # (skip names with special chars, very short names, or very long names)
    _SKIP_CHARS = re.compile(r'[,()/\[\]+]')
    for name in registry_names:
        clean = str(name).lower().strip()
        if len(clean) < 7 or len(clean) > 40 or _SKIP_CHARS.search(clean):
            continue
        for n in (5, 6):
            combined.setdefault(clean[:n], str(name))

    # Longest keys first for greedy prefix matching
    sorted_items = tuple(sorted(combined.items(), key=lambda x: len(x[0]), reverse=True))
    return sorted_items


def abbrev_lookup(clean_token: str, abbrev_items: tuple) -> str | None:
    """Returns canonical name if token matches a known abbreviation, else None.

    CRITICAL FIX: Short tokens (≤4 chars) use EXACT match only.
    The abbreviations.json has 29,000+ entries including many 3-4 char codes
    that are prefixes of longer drug names. Using prefix-regex on short tokens
    causes 'met' to match 'metfo' → 'Metaraminol for' before reaching the
    correct 'met' → 'Metformin' entry. Exact match prevents this.
    """
    t = clean_token.lower().strip()
    # For short tokens, require exact match (no prefix expansion)
    if len(t) <= 4:
        for abbrev, full_name in abbrev_items:
            if abbrev.lower().strip() == t:
                return full_name
        return None
    # For longer tokens, use prefix regex as before
    for abbrev, full_name in abbrev_items:
        if re.match(r'^' + re.escape(abbrev.lower().strip()) + r'(\s|/|$)', t, re.IGNORECASE):
            return full_name
    return None


def abbrev_lookup_fallback(clean_token: str, abbrev_items: tuple, master_df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Try abbreviation lookup; if resolved name not in registry, find closest TF-IDF match.
    Returns (resolved_name, fallback_used).

    CRITICAL FIX: If the abbreviation resolves to a known _PHARMA_OVERRIDE target
    (e.g. 'met'→'Metformin', 'para'→'Paracetamol'), return it DIRECTLY without
    registry validation. Previously, if the registry cache was stale and didn't
    contain 'Metformin' yet, the function fell through to TF-IDF which then
    picked the closest match ('Formic Acid') — completely wrong.
    """
    abbrev_result = abbrev_lookup(clean_token, abbrev_items)
    if not abbrev_result:
        return None, None

    # FAST PATH: known pharma override targets are always trusted — no registry check needed
    if abbrev_result.lower() in _PHARMA_OVERRIDE_TARGETS:
        return abbrev_result, None

    if master_df.empty:
        return None, None

    # Check if the resolved abbreviation exists in registry
    registry_names = master_df["generic_name"].astype(str).str.lower().str.strip().tolist()
    if abbrev_result.lower() in registry_names:
        return abbrev_result, None  # Found directly

    # Name not in registry — try fuzzy match on the abbreviation-resolved name
    # Only do this for non-pharma-override results to avoid wrong fallbacks
    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(registry_names)
        query = vectorizer.transform([abbrev_result.lower()])
        scores = cosine_similarity(query, matrix)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= 0.55:  # Raised threshold: only accept high-confidence fallbacks
            return master_df.iloc[best_idx]["generic_name"], f"abbrev→{abbrev_result}"
    except Exception:
        pass

    return None, None

# =====================================================================
# 7. PHASE 3 — LAYER 2: TF-IDF COSINE SIMILARITY
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

    BUG FIX (Performance): The caller previously passed a freshly-built
    pool_list (generic + brand concatenated) as the tuple, which changed
    every call when brand_name values differed slightly, busting the cache.
    Now the caller passes only generic_names (stable) and we build the
    combined pool inside this cached function so the cache key is stable.
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
    """Batch TF-IDF match — one matrix multiply for all unresolved rows.

    BUG FIX (Performance): Pass only generic_names tuple as cache key so
    build_tfidf_index cache is stable across repeated calls in the same session.
    Previously the pool_list included brand_name which varied and busted cache.
    """
    n = len(clean_tokens)
    if master_df.empty or n == 0:
        return [(None, 0.0)] * n

    # Use only generic names for the TF-IDF index (stable cache key)
    generic_names_tuple = tuple(master_df["generic_name"].astype(str).tolist())

    try:
        vectorizer, matrix = build_tfidf_index(region_key, generic_names_tuple)
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
# 8. PHASE 4 — LAYER 3: GEMINI AI FALLBACK
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
# 9. RECONCILIATION ENGINE
# =====================================================================
def run_reconciliation(client_df: pd.DataFrame, master_df: pd.DataFrame,
                       jurisdiction: str, l2_threshold: float,
                       use_ai: bool, verbose: bool = False) -> pd.DataFrame:
    """Full pipeline per invoice row"""
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
    l2_tried       = [False] * n_rows
    l2_best_score  = [0.0] * n_rows
    cache_dirty    = False
    registry_set   = set(registry_names)

    diagnostics    = [{"step": "init"} for _ in range(n_rows)] if verbose else None

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

        if verbose:
            diagnostics[i]["raw"] = raw_inputs[i]
            diagnostics[i]["clean"] = clean_tokens[i]
            diagnostics[i]["layer1_attempts"] = []

        cached = cache_db.get(hash_keys[i]) or st.session_state["ai_cache"].get(hash_keys[i])
        if cached:
            cached_name = cached.get("generic_name", "")
            if cached_name in registry_set and _cache_is_plausible(clean_tokens[i], cached_name):
                resolved[i]     = cached_name
                brand_names[i]  = cached.get("brand_name", "N/A")
                system_ids[i]   = cached.get("system_id", "N/A")
                match_layers[i] = "Layer 1: Cache Hit"
                confidences[i]  = 1.0
                if verbose:
                    diagnostics[i]["layer1_attempts"].append(f"Cache: HIT -> {cached_name}")
                continue

        if verbose:
            diagnostics[i]["layer1_attempts"].append("Cache: miss")

        abbrev_name, fallback_note = abbrev_lookup_fallback(clean_tokens[i], abbrev_items, master_df)
        if abbrev_name:
            name, reg_idx = resolve_in_registry(abbrev_name, master_df, registry_index, region_key, from_abbrev=True)
            if name:
                resolved[i]     = name
                match_layers[i] = "Layer 1: Abbreviation Lookup"
                confidences[i]  = 1.0
                brand_names[i]  = str(master_df.iloc[reg_idx]["brand_name"])
                system_ids[i]   = str(master_df.iloc[reg_idx]["system_id"])
                if verbose:
                    diagnostics[i]["layer1_attempts"].append(f"Abbrev: '{abbrev_name}' -> {name}")
                continue

        name, reg_idx = resolve_in_registry(clean_tokens[i], master_df, registry_index, region_key, from_abbrev=False)
        if name:
            resolved[i]     = name
            match_layers[i] = "Layer 1: Exact Name Match"
            confidences[i]  = 1.0
            brand_names[i]  = str(master_df.iloc[reg_idx]["brand_name"])
            system_ids[i]   = str(master_df.iloc[reg_idx]["system_id"])
            if verbose:
                diagnostics[i]["layer1_attempts"].append(f"Exact: HIT -> {name}")
            continue

    l2_pending = [i for i in range(n_rows) if not resolved[i]]
    if l2_pending:
        batch_tokens = [clean_tokens[i] for i in l2_pending]
        l2_threshold_actual = max(0.30, l2_threshold * 0.8)
        batch_hits   = tfidf_match_batch(batch_tokens, master_df, region_key, l2_threshold_actual)
        for i, (tfidf_name, tfidf_score) in zip(l2_pending, batch_hits):
            l2_tried[i]      = True
            l2_best_score[i] = tfidf_score
            if tfidf_name and tfidf_score >= l2_threshold_actual:
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

    rows = []
    for i in range(n_rows):
        resolved_name = resolved[i]
        if not resolved_name:
            resolved_name = "UNRESOLVED — HUMAN OVERRIDE REQUIRED"
            if l2_tried[i] and l2_best_score[i] > 0:
                match_layers[i] = f"Layer 2: Below Threshold ({l2_best_score[i]:.1%})"
                confidences[i] = l2_best_score[i]
            elif not l2_tried[i]:
                match_layers[i] = "Layer 1: No Match"
                confidences[i] = 0.0
            baseline_price = 0.0
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

    if verbose and diagnostics:
        st.session_state["diagnostics"] = diagnostics

    return pd.DataFrame(rows)

# =====================================================================
# 10. DISPLAY HELPER
# =====================================================================
def style_rows(row):
    # Use .get() to safely check for multiple possible column names
    match_layer = str(row.get("Match Layer", ""))
    generic_name = str(row.get("Generic Name", row.get("generic_name", row.get("VMP Name", ""))))
    
    if "Flagged" in match_layer or "UNRESOLVED" in generic_name:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

# =====================================================================
# 11. MAIN UI - LANDING PAGE, LOGIN, AND APP ENGINE
# =====================================================================
# STATE 1: LANDING PAGE
if st.session_state.page == "landing":
    nav_col1, nav_col2 = st.columns([8, 2])
    with nav_col1:
        st.markdown("### 🧬 **APEX LOGIC**")
    with nav_col2:
        if st.button("🔒 Client Login Portal", use_container_width=True, type="secondary"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("---")
    hero_col1, hero_col2 = st.columns([6, 4])
    with hero_col1:
        st.markdown('<p class="hero-title">Stop Manually Wrestling with <span class="accent-text">Pharmacy Vendor Invoices</span>.</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-hero">The automated data harmonization engine built for modern pharmacies. Instantly scrub, standardize, and format messy supplier spreadsheets across the UK, Canada, and the US.</p>', unsafe_allow_html=True)
    with hero_col2:
        st.info("📦 **Data Pipeline Ingestion Stream**\n\n→ `PCM 500mg (UK)` *(Messy Vendor Invoiced Shorthand)*\n\n⚡ *Apex Harmonization Matrix Processing...*\n\n→ **Paracetamol 500mg Tablet (Active Canonical ID: 104)**")

    st.markdown("##")
    st.markdown("---")
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
    st.warning("🚀 **Global Pilot Group Initiative (Strictly Limited)**: To seed our foundational pilot group across our launch regions, We are offering an exclusive **50% lifetime discount** to the first **5 pharmacies** to register in the UK, Canada, and the US. Use code **`APEXEARLY50`** at checkout. Once the 5th slot in your region is filled, the system automatically reverts to regular pricing.")

    st.markdown("### 💳 Corporate Deployment Packages")
    tier1_col, tier2_col, tier3_col = st.columns(3)
    with tier1_col:
        st.markdown('<div class="price-card"><h4>📦 Tier 1: Standalone Toolkit</h4><h2>£99 <span style="font-size:1rem;color:#A0AEC0">One-Time Payment</span></h2><hr style="border-color:#2D3748"><p style="margin-bottom:0.5rem;">• Full master abbreviation mapping package (abbreviations.json)</p><p style="margin-bottom:0.5rem;">• Local pipeline implementation and audit templates</p><p style="margin-bottom:0.5rem;">• Optimized for internal development and localized audit setups</p></div>', unsafe_allow_html=True)
        st.link_button("Get Standalone Kit", "https://whop.com/your-tier-1-checkout-link", use_container_width=True)
    with tier2_col:
        st.markdown('<div class="price-card-premium"><h4>🧬 Tier 2: Cloud Instance Software Access</h4><h2>£499 <span style="font-size:1rem;color:#A0AEC0">/ Month</span></h2><hr style="border-color:#00FFBB"><p style="margin-bottom:0.5rem;">• Infinite, automated 24/7 web application invoice uploads</p><p style="margin-bottom:0.5rem;">• Multi-regional database compliance (UK, Canada, US formats)</p><p style="margin-bottom:0.5rem;">• Absolute cross-border matrix validation engine</p></div>', unsafe_allow_html=True)
        st.link_button("Launch Cloud Engine", "https://whop.com/your-tier-2-checkout-link", type="primary", use_container_width=True)
    with tier3_col:
        st.markdown('<div class="price-card"><h4>🏢 Tier 3: Bespoke Enterprise Integration</h4><h2>£2,500 <span style="font-size:1rem;color:#A0AEC0">Setup</span> + £250<span style="font-size:1rem;color:#A0AEC0">/Mo</span></h2><hr style="border-color:#2D3748"><p style="margin-bottom:0.5rem;">• Custom logic mapped directly to your specific PMS layouts</p><p style="margin-bottom:0.5rem;">• Secure automated webhooks and pipeline scripts</p><p style="margin-bottom:0.5rem;">• Priority server allocation & dedicated architecture kickoff strategy call</p></div>', unsafe_allow_html=True)
        st.link_button("Book Integration Call", "https://calendly.com/your-link", use_container_width=True)

# STATE 2: LOGIN PAGE
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

# STATE 3: PROTECTED APP ENGINE
if st.session_state.authenticated or st.session_state.page == "app":
    with st.sidebar:
        st.markdown("### 📊 Parameters")
        jurisdiction = st.selectbox("Region", ["UK", "US", "Canada"])
        l2_threshold = st.slider("Layer 2 Confidence Threshold", 0.30, 1.00, 0.50, help="TF-IDF cosine scores for pharmacy names realistically peak at 0.5–0.8.")
        use_ai = st.toggle("Enable Layer 3 AI Fallback", value=False, help="Requires GEMINI_API_KEY in Streamlit secrets.")
        if use_ai and not st.secrets.get("GEMINI_API_KEY", ""):
            st.sidebar.error("⚠️ GEMINI_API_KEY missing from secrets. Layer 3 will not fire.")
        enable_diagnostics = st.toggle("Show Diagnostic Output", value=False, help="Display layer-by-layer resolution details for first 20 rows.")
        if st.button("🗑️ Clear Resolution Cache", help="Remove stale/wrong cached drug mappings from prior runs"):
            clear_cache_db()
            st.session_state.pop("run_id", None)
            st.success("Cache cleared — re-upload invoice to re-run pipeline.")
            st.rerun()
        st.write("---")
        st.caption(f"Region: {jurisdiction} | L2 threshold: {l2_threshold:.0%}")
        st.write("---")
        with st.container(border=True):
            st.markdown("<small>🔒 **ARCHITECTURAL BOUNDARY**</small>", unsafe_allow_html=True)
            st.caption("No EMR/EHR or eRx connectivity. Financial reconciliation only.")
        st.markdown("### 🏛️ Enterprise Support")
        st.link_button("💻 Service Desk", url="https://support.apexlogic.ai/portal", use_container_width=True)
        st.link_button("📞 Priority Callback", url="mailto:enterprise-ops@apexlogic.ai?subject=URGENT", use_container_width=True)
        if st.button("🚪 Secure Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.page = "landing"
            st.rerun()

    st.markdown('<div class="brand-header">🧬 APEX LOGIC</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">GLOBAL PHARMACY AUDIT ENGINE</div>', unsafe_allow_html=True)

    tab_overview, tab_workspace = st.tabs(["🏠 Platform Overview", "⚡ Automated Audit Suite"])

    with tab_overview:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### Reclaim Your Pharmacy's Lost Margin")
            st.write("Apex Logic processes distributor invoices through a three-layer resolution engine — deterministic abbreviation lookup, TF-IDF cosine similarity, and AI fallback — then cross-references every resolved drug against live national baseline registries to surface hidden price gouging before you clear accounts payable.")
            st.markdown("#### 🌍 Regions Supported")
            st.info("**United Kingdom:** NHS dm+d\n\n**United States:** FDA NDC\n\n**Canada:** Health Canada DPD")
            st.markdown("#### ⚙️ Three-Layer Pipeline")
            st.markdown("| Layer | Method | Example |\n|---|---|---|\n| **Layer 1a** | Hash cache (persistent) | Same supplier + drug → instant hit |\n| **Layer 1b** | Abbreviation dictionary (150+ shortcuts) | `MET → Metformin`, `PCM → Paracetamol` |\n| **Layer 1c** | Exact name match after noise strip | `LISINOPRIL 5MG TABS` → `Lisinopril` |\n| **Layer 2** | TF-IDF char n-gram cosine similarity | `SERTRALINE` → `Sertraline` |\n| **Layer 3** | Gemini AI (bounded JSON schema) | Anything ambiguous |")
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
        master_registry = load_jurisdictional_registry(jurisdiction)
        j_profile       = JURISDICTION_PROFILES[jurisdiction]
        st.markdown("#### 📑 Invoice Upload")
        uploaded_file = st.file_uploader("Upload invoice (.csv or .xlsx)", type=["csv", "xlsx"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                client_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"🚨 Could not read file: {e}")
                st.stop()

            st.markdown("##### Raw Invoice Preview")
            st.dataframe(client_data.head(5), use_container_width=True)

            raw_columns    = client_data.columns.tolist()
            normalized_map = {c: c.strip().lower().replace("_","").replace(" ","").replace("-","").replace("/","") for c in raw_columns}
            drug_synonyms  = ["drugname","drug","product","productname","item","itemdescription","medication","medicine","description","standardname","molecule","activeingredient","clinicalname","brand","clientdrugname"]
            price_synonyms = ["unitprice","price","cost","currentprice","rate","amount","procurementcost","contractprice","billingamount","acquisitioncost","invoiceprice","clientcurrentprice"]
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
            working_df["Client_Current_Price"] = (client_data[price_col].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0").astype(float))
            sku_col = next((c for c in raw_columns if normalized_map[c] in ("sku", "vendorsku", "distributorsku", "productcode", "itemcode")), None)
            if sku_col:
                working_df["SKU"] = client_data[sku_col].astype(str)
            st.toast("✅ Invoice ingested — running pipeline...", icon="⚡")

            ENGINE_VERSION = "v6"
            ai_flag        = "ai_on" if use_ai else "ai_off"
            run_id         = f"{ENGINE_VERSION}_{uploaded_file.name}_{jurisdiction}_{l2_threshold}_{ai_flag}"

            if st.session_state.get("run_id") != run_id:
                with st.spinner("Running three-layer pipeline..."):
                    st.session_state["audit_data"] = run_reconciliation(working_df, master_registry, jurisdiction, l2_threshold, use_ai, verbose=enable_diagnostics)
                    st.session_state["run_id"] = run_id

            results_df = st.session_state["audit_data"]
            overcharges  = results_df[results_df["Audit Verdict"] == "🚨 TARIFF OVERCHARGE"]
            unresolved   = results_df[results_df["Generic Name"].str.contains("UNRESOLVED")]
            l1_hits      = results_df[results_df["Match Layer"].str.startswith("Layer 1")]
            l2_hits      = results_df[results_df["Match Layer"].str.startswith("Layer 2: TF-IDF")]
            l2_miss      = results_df[results_df["Match Layer"].str.startswith("Layer 2: Below")]
            l3_ai_hits   = results_df[results_df["Match Layer"] == "Layer 3: AI Resolution"]
            exposure     = overcharges["Price Variance"].sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Audited", len(results_df))
            m2.metric("Overcharge Exposure", f"${exposure:,.2f}", delta=f"{len(overcharges)} lines", delta_color="inverse")
            m3.metric("Unresolved", len(unresolved))
            m4.metric("Auto-Resolved", f"{len(results_df) - len(unresolved)}/{len(results_df)}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("⚡ Layer 1 (Cache + Lookup)", len(l1_hits))
            r2.metric("🔬 Layer 2 (TF-IDF Hit)", len(l2_hits))
            r3.metric("📉 Layer 2 (Below Threshold)", len(l2_miss))
            r4.metric("🤖 Layer 3 (AI)", len(l3_ai_hits))

            display_df = results_df.drop(columns=["_raw_score", "_hash"], errors="ignore").copy()
            display_df = display_df.rename(columns={"Generic Name": j_profile["generic_label"], "Brand Name": j_profile["brand_label"], "System ID": j_profile["id_label"]})
            for col in ["Invoice Price", "Registry Baseline", "Price Variance"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map("${:,.4f}".format)
            if "Variance %" in display_df.columns:
                display_df["Variance %"] = display_df["Variance %"].map("{:,.1f}%".format)

            st.markdown("##### Audit Results")
            st.dataframe(display_df.style.apply(style_rows, axis=1), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📤 Export")
            ex1, ex2 = st.columns(2)
            with ex1:
                clean_cols = ["SKU", j_profile["generic_label"], j_profile["brand_label"], "Invoice Price", "Registry Baseline", "Price Variance", "Audit Verdict"]
                clean_export = display_df[[c for c in clean_cols if c in display_df.columns]]
                st.download_button("📦 Clean PMS File", data=clean_export.to_csv(index=False).encode(), file_name=f"apex_pms_{jurisdiction.lower()}.csv", mime="text/csv")
            with ex2:
                st.download_button("🔎 Full Audit Workbook", data=display_df.to_csv(index=False).encode(), file_name=f"apex_audit_{jurisdiction.lower()}.csv", mime="text/csv")
        else:
            st.write("---")
            st.info("Upload an invoice to begin.")

