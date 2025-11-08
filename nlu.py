# nlu.py - Natural Language Understanding Layer
# Responsibility: Intent classification + Slot filling ONLY
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer, util

# ========== MODEL INITIALIZATION ==========
_ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========== INTENT PROTOTYPES ==========
INTENT_PROTOTYPES = {
    # RAG: textual guidance / procedures / standards
    "RAG": [
        "What are the mowing steps and safety requirements?",
        "How do I maintain the turf properly?",
        "What equipment do I need for mowing?",
        "Tell me the standard operating procedures for mowing",
        "What are the dimensions for U15 soccer?",
        "Show me baseball field requirements for U13",
        "What's the pitching distance for female softball U17?",
    ],

    # SQL_tool: structured results only (aggregations/ranking/lookup), no explanation
    "SQL_tool": [
        "How did total mowing costs trend from April to June 2025?",
        "Compare mowing costs across all parks in March 2025",
        "When was the last mowing at Cambridge Park?",
        "Show me all parks ranked by mowing cost",

        # Moved from RAG+SQL_tool → SQL_tool (pure aggregation/ranking semantics)
        "Which park had the highest total mowing labor cost in March 2025?",
        "Show me the most expensive park for mowing in April",
        "What was the top mowing cost by location last month?",
    ],

    # RAG+SQL_tool: data + interpretation/grounding in mowing standards or policy
    "RAG+SQL_tool": [
        "Explain why mowing labor cost was highest in March 2025 based on mowing guidelines.",
        "Interpret April mowing expenses using standard mowing practices.",
        "Discuss whether last month’s top mowing cost aligns with mowing policy thresholds.",
        "Explain anomalies in the 2024 mowing cost trend with reference to recommended maintenance frequency.",
        "Compare irrigation repair costs in Q2 2025 and cite relevant maintenance standards.",
        "Discuss reasons for above-average mowing cost in June using mowing procedures.",
        "Interpret median mowing cost per park in Q3 2024 with reference to mowing standards.",
        "Explain July YOY change in mowing costs using recommended mowing intervals.",
        "Calculate the differences in dimensions of a [shape] field for [sport]",
        "What are the differences for the diamond fields for Softball Female - U17",
        "What are the differences for the rectangular fields for U12 and U14?"
    ],

    # RAG+CV_tool: image + textual knowledge (needs references/explanations)
    "RAG+CV_tool": [
        "Here is a picture. Estimate turf repair effort and reference the recommended mowing procedures.",
        "Check this image and tell me if the grass needs mowing with references to the mowing guideline.",
        "Analyze this field photo and recommend actions, citing the relevant mowing standard.",
    ],

    # CV_tool: pure visual judgment (no textual citation)
    "CV_tool": [
        "Check this image and assess turf condition.",
        "Analyze this photo of the field.",
        "What do you see in this image?",
    ],
}

# ===== Encode prototypes (keep this right after INTENT_PROTOTYPES) =====
_PROT_TEXTS = []
_PROT_LABELS = []
for label, samples in INTENT_PROTOTYPES.items():
    for sample in samples:
        _PROT_TEXTS.append(sample)
        _PROT_LABELS.append(label)
_PROT_EMB = _ENCODER.encode(_PROT_TEXTS, normalize_embeddings=True)

# ========= Keyword cue lists for refinement =========
_SQL_ONLY_CUES = [
    "sql:", "duckdb", "select ", " from ", " where ",
    "group by", "table only", "numbers only", "no explanation",
    "rows only", "csv", "data frame only", "return table",
    "pivot table", "counts only", "scalar", "integer only",
    "series output", "quartiles", "two columns", "list names only"
]

# SQL explain/interpretation cues (push to RAG+SQL_tool)
_SQL_EXPLAIN_CUES = [
    "explain", "interpret", "why", "reason", "rationale", "context",
    "with notes", "with reference", "with references", "cite", "citation",
    "commentary", "insights", "insight"
]

# General analysis cues that usually imply a structured SQL answer,
# but do not explicitly request explanations.
_SQL_GENERAL_CUES = [
    "trend", "compare", "median", "top", "highest", "max",
    "yoy", "year over year", "last", "recent", "rank", "breakdown"
]

# Informational textual guidance cues (only used when not SQL-intent)
_INFORMATIONAL_KEYWORDS = [
    "steps", "procedure", "safety", "manual", "how to", "sop",
    "dimensions", "requirements", "standards", "what are", "show me", "tell me"
]

# Extra policy/standard cues that imply explanatory grounding (all lowercase)
_RAG_POLICY_CUES = [
    "standard", "standards", "guideline", "guidelines",
    "policy", "policies", "threshold", "thresholds",
    "interval", "intervals", "frequency", "frequencies",
    "requirement", "requirements"
]

# Text mentions that imply the user is talking about or expecting an image
_CV_TEXT_CUES = ["image", "photo", "picture", "screenshot", "attached", "see image", "see pic"]

# Broad data cues: words that commonly indicate a structured/quantitative ask
_DATA_CUES = [
    "cost", "costs", "hours", "count", "counts", "visits", "median", "mean", "average",
    "total", "sum", "trend", "compare", "top", "highest", "max", "rank", "last", "date",
    "per park", "by park", "per month", "by month", "yoy", "year over year", "histogram",
    "table", "csv", "series"
]
# ========== ENTITY EXTRACTION UTILITIES ==========
_MONTHS = {
    m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"],
        start=1
    )
}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}


def _normalize(text: Optional[str]) -> str:
    """Normalize text to lowercase and strip whitespace."""
    return (text or "").strip().lower()


def _parse_month_year(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract single month and year from text."""
    t = _normalize(text).replace(",", " ")
    # Pattern: YYYY-MM
    m = re.search(r"\b(20\d{2})-(\d{1,2})\b", t)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return month, year
    # Full month name
    for name, idx in _MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, year
    # Abbrev month
    for abbr, idx in _MONTH_ABBR.items():
        if re.search(rf"\b{abbr}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, year
    # Year only
    year_match = re.search(r"\b(20\d{2})\b", t)
    if year_match:
        return None, int(year_match.group(1))
    return None, None


def _parse_month_range(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract month range from text.
    Returns: (start_month, end_month, year)
    """
    t = _normalize(text)
    # Pattern: "from April to June 2025" / "between Apr and Jun 2025"
    m = re.search(
        r"(?:from|between)\s+([a-zA-Z]+)\s+(?:to|and)\s+([a-zA-Z]+)\s*(20\d{2})?",
        t
    )
    if m:
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        year = int(m.group(3)) if m.group(3) else None

        def to_month(s: str) -> Optional[int]:
            if s in _MONTHS:
                return _MONTHS[s]
            return _MONTH_ABBR.get(s[:3])

        return to_month(m1), to_month(m2), year

    # Pattern: "in June 2025"
    for name, idx in _MONTHS.items():
        if re.search(rf"\bin\s+{name}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, idx, year
    for abbr, idx in _MONTH_ABBR.items():
        if re.search(rf"\bin\s+{abbr}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, idx, year
    return None, None, None


def _parse_park_name(text: str) -> Optional[str]:
    """Extract park name from text (very lightweight heuristic)."""
    t = _normalize(text)
    # Pattern: "in/at/for XXX Park"
    m = re.search(r"(?:in|at|for)\s+([a-z][a-z\s\-\&]+(?:park|pk))\b", t)
    if m:
        park_raw = m.group(1).strip()
        park_clean = park_raw.replace(" park", "").replace(" pk", "").strip()
        return park_clean.title()
    # Known examples (extend as needed)
    known_parks = [
        "alice town", "cambridge", "garden", "grandview",
        "mcgill", "mcspadden", "mosaic creek", "cariboo",
        "john hendry", "hastings", "new brighton"
    ]
    for park in known_parks:
        if park in t:
            return park.title()
    return None

def _parse_sport(text: str) -> Optional[Dict[str, Any]]:
    """Detect sport, gender, age groups, and optional format/category from a query.

    Returns a dict:
      {
        "sport": str,            # normalized sport name (e.g. "soccer", "softball", "baseball")
        "gender": str|None,      # "male", "female", "mixed", or None
        "age_groups": List[str], # e.g. ["U5","U6","U12"]
        "format": str|None,      # e.g. "9v9", "11v11", "oval infield", "slow-pitch"
        "category": str|None     # e.g. "Masters", "A", "B", "C", "D"
      }
    """
    if not text:
        return None

    t = _normalize(text)

    # --- sport detection (priority order) ---
    sport = None
    if "soccer" in t or "soccer/" in t:
        sport = "soccer"
    elif "gaelic" in t and "football" in t:
        sport = "gaelic football"
    elif "cricket" in t:
        sport = "cricket"
    elif "ultimate" in t or "frisbee" in t:
        sport = "ultimate frisbee"
    elif "lacrosse" in t:
        sport = "lacrosse"
    elif "rugby" in t:
        sport = "rugby"
    elif "softball" in t or "sofball" in t or "softbal" in t:
        sport = "softball"
    elif ("slo" in t and "pitch" in t) or "slow-pitch" in t or "slow pitch" in t:
        sport = "softball"
    elif "baseball" in t:
        sport = "baseball"
    elif "cfl" in t or "nfl" in t:
        sport = "gridiron"
    elif "football" in t:
        # prefer soccer if keyword appears elsewhere; otherwise mark generic football
        sport = "football"
    else:
        # fallback: try to find any of the known keywords
        for k, name in [
            ("soccer", "soccer"), ("softball", "softball"), ("baseball", "baseball"),
            ("cricket", "cricket"), ("lacrosse", "lacrosse"), ("rugby", "rugby"),
            ("ultimate", "ultimate frisbee"), ("frisbee", "ultimate frisbee")
        ]:
            if k in t:
                sport = name
                break

    # --- gender detection ---
    gender = None
    if re.search(r"\b(male|men|mens|boys)\b", t):
        gender = "male"
    elif re.search(r"\b(female|women|womens|ladies|girls)\b", t):
        gender = "female"
    elif re.search(r"\b(mixed|co-?ed|coed)\b", t):
        gender = "mixed"
    elif "masters" in t:
        # masters often implies adult; gender may be specified elsewhere
        gender = gender or None

    # --- age group extraction (U# patterns) ---
    age_raw = re.findall(r"\bU\s*[-]?\s*\d{1,2}\b", t, flags=re.IGNORECASE)
    age_groups = []
    for a in age_raw:
        # normalize "U 12" / "U-12" / "U12" -> "U12"
        m = re.search(r"U\s*-?\s*(\d{1,2})", a, flags=re.IGNORECASE)
        if m:
            age_groups.append(f"U{int(m.group(1))}")

    # also capture patterns like "U10 & U11" (handled above) and ranges like "U8-U9"
    # de-duplicate while preserving order
    seen = set()
    age_groups = [x for x in age_groups if not (x in seen or seen.add(x))]

    # --- format detection (e.g. 9v9, 11v11, oval infield) ---
    fmt = None
    vf = re.search(r"\b(\d+)\s*v\s*(\d+)\b", t)
    if vf:
        fmt = f"{vf.group(1)}v{vf.group(2)}"
    elif "9v9" in t or "9 v 9" in t:
        fmt = "9v9"
    elif "11v11" in t or "11 v 11" in t:
        fmt = "11v11"
    elif "oval" in t and "infield" in t:
        fmt = "oval infield"
    elif ("slow-pitch" in t) or ("slow pitch" in t) or ("slo pitch" in t) or ("slo - pitch" in t):
        fmt = "slow-pitch"

    # --- category detection (Masters, A/B/C/D etc.) ---
    category = None
    if "masters" in t:
        category = "Masters"
    # capture letter grades like "A,B & C" or single letter categories
    letters = re.findall(r"\b([A-D])\b", text.upper())
    if letters:
        category = ",".join(sorted(set(letters), key=letters.index))

    # If no sport found, return None
    if not sport:
        return None

    return {
        "domain": "field_dimension",
        "sport": sport,
        "gender": gender,
        "age_groups": age_groups,
        "format": fmt,
        "category": category
    }

def _parse_cross_year_month_range(text: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract month range that may span multiple years from text.
    Specifically handles patterns like "from June 2024 to May 2025".
    
    Returns: (start_month, end_month, start_year, end_year)
    
    Examples:
        "from June 2024 to May 2025" -> (6, 5, 2024, 2025)
        "between Apr 2023 and Jun 2024" -> (4, 6, 2023, 2024)
    """
    t = _normalize(text)
    
    # Pattern: "from/between [Month] [Year] to/and [Month] [Year]"
    pattern = r"(?:from|between)\s+([a-zA-Z]+)\s+(20\d{2})\s+(?:to|and)\s+([a-zA-Z]+)\s+(20\d{2})"
    m = re.search(pattern, t)
    
    if not m:
        return None, None, None, None
    
    month1_str = m.group(1).lower()
    year1 = int(m.group(2))
    month2_str = m.group(3).lower()
    year2 = int(m.group(4))
    
    def to_month(s: str) -> Optional[int]:
        """Convert month name/abbreviation to month number (1-12)."""
        if s in _MONTHS:
            return _MONTHS[s]
        return _MONTH_ABBR.get(s[:3])
    
    month1 = to_month(month1_str)
    month2 = to_month(month2_str)
    
    if month1 is None or month2 is None:
        return None, None, None, None
    
    return month1, month2, year1, year2

def _detect_domain(text: str) -> str:
    """Detect query domain: mowing / field_standards / generic."""
    t = _normalize(text)
    if any(k in t for k in ["mowing", "mow", "turf", "grass", "lawn"]):
        return "mowing"
    elif any(k in t for k in ["dimension", "fields", "field", "dimensions"]):
        return "field_dimension"
    if any(k in t for k in ["soccer", "baseball", "softball", "cricket",
                            "football", "rugby", "field", "dimensions",
                            "pitching", "u10", "u11", "u12", "u13",
                            "u14", "u15", "u16", "u17", "u18"]):
        return "field_standards"
    return "generic"


# ========== NLU RESULT DATACLASS ==========
@dataclass
class NLUResult:
    """Output from NLU layer."""
    intent: str
    confidence: float
    slots: Dict[str, Any]
    raw_query: str


# ========== MAIN NLU FUNCTION ==========
def parse_intent_and_slots(
    text: str,
    image_uri: Optional[str] = None
) -> NLUResult:
    """
    Parse user input to extract intent and slots.

    Args:
        text: User query text
        image_uri: Optional image URI

    Returns:
        NLUResult with intent classification and extracted entities
    """
    query = (text or "").strip()
    lowq = _normalize(query)

    # ----- STEP 1: Prototype-based initial intent -----
    query_embedding = _ENCODER.encode([query], normalize_embeddings=True)
    similarities = util.cos_sim(query_embedding, _PROT_EMB).cpu().tolist()[0]
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    intent = _PROT_LABELS[best_idx]
    confidence = float(similarities[best_idx])
    print(f"[NLU] Initial intent: {intent} (confidence: {confidence:.3f})")

    # ----- Enhancement A: Low-confidence fallback -----
    LOW_CONF = 0.35  # tune 0.30~0.45 based on evaluation
    if confidence < LOW_CONF:
        intent = "RAG+CV_tool" if image_uri else "RAG"
        print(f"[NLU] Low confidence ({confidence:.3f}) → fallback to {intent}")

    # ----- STEP 2: SQL/CV routing with combined cues and image awareness -----
    explanation_requested = False

    # Cue booleans
    has_sql_only    = any(k in lowq for k in _SQL_ONLY_CUES)
    has_sql_general = any(k in lowq for k in _SQL_GENERAL_CUES)
    has_data_cues   = any(k in lowq for k in _DATA_CUES)
    has_sql_query   = has_sql_only or has_sql_general or has_data_cues

    has_explain_kw  = any(k in lowq for k in _SQL_EXPLAIN_CUES)
    has_info_kw     = any(k in lowq for k in _INFORMATIONAL_KEYWORDS)
    has_policy_kw   = any(k in lowq for k in _RAG_POLICY_CUES)

    has_image_flag  = bool(image_uri)
    mentions_image  = any(k in lowq for k in _CV_TEXT_CUES)
    cv_context      = has_image_flag or mentions_image

    # 1) Hard SQL-only always wins: explicit table/SQL syntax requests
    if has_sql_only:
        intent = "SQL_tool"
        explanation_requested = False
        print("[NLU] SQL-only cues detected → intent=SQL_tool")

    # 2) If user is in a CV context (image present or text mentions an image)
    #    then prefer CV path unless it's truly SQL-only.
    elif cv_context:
        # If the ask also mixes data analysis with explanations/policy, keep it multimodal
        if has_sql_query and (has_explain_kw or has_policy_kw or has_info_kw):
            intent = "RAG+CV_tool"
            explanation_requested = True
            print("[NLU] CV context + (data+explain/policy/info) → intent=RAG+CV_tool")
        else:
            # otherwise leave as-is (RAG) and let Step 4 upgrade to RAG+CV_tool
            print("[NLU] CV context detected; defer to Step 4 for potential upgrade")

    # 3) No CV context. Combine SQL signals with explanatory/policy intent → RAG+SQL_tool
    elif has_sql_query and (has_explain_kw or has_policy_kw or has_info_kw):
        intent = "RAG+SQL_tool"
        explanation_requested = True
        print("[NLU] SQL + (policy/info/explain) detected → intent=RAG+SQL_tool")

    # 4) Plain SQL-general without explanation/policy → SQL_tool
    elif has_sql_general:
        intent = "SQL_tool"
        print("[NLU] SQL-general cues detected → intent=SQL_tool")


    # ----- STEP 3: Informational fallback (only if not SQL) -----
    if intent not in ("SQL_tool", "RAG+SQL_tool"):
        if any(k in lowq for k in _INFORMATIONAL_KEYWORDS):
            # Prefer RAG for informational queries (textual guidance).
            intent = "RAG"
            print("[NLU] Informational keywords detected → intent=RAG")

    # ----- STEP 4: CV-aware adjustment (conservative) -----
    # If an image exists and current intent is RAG, upgrade to RAG+CV_tool.
    if image_uri:
        if intent == "RAG":
            intent = "RAG+CV_tool"
            print("[NLU] Image detected; upgrading RAG → RAG+CV_tool")
        # If intent is SQL_tool / RAG+SQL_tool, do NOT force to CV; keep SQL path.
    else:
        # No image but predicted a CV intent → downgrade to RAG (textual fallback).
        if "CV" in intent:
            print("[NLU] No image but CV intent detected; downgrading to RAG")
            intent = "RAG"

    # ----- STEP 5: Slot Filling (NER-lite) -----
    domain = _detect_domain(query)
    if domain == "field_dimension":
        sport_info = _parse_sport(query)
        return NLUResult(
            intent=intent,
            confidence=round(confidence, 3),
            slots=sport_info or {},
            raw_query=query
        )
    month, year = _parse_month_year(query)
    start_month, end_month, range_year = _parse_month_range(query)
    park_name = _parse_park_name(query)
    month1, month2, year1, year2 = _parse_cross_year_month_range(query) # least cost for a period

    slots = {
        "domain": domain,
        "month": month,
        "year": year,
        "start_month": start_month,
        "end_month": end_month,
        "range_year": range_year,
        "month1": month1, # least cost for a period
        "month2": month2, # least cost for a period
        "year1": year1, # least cost for a period
        "year2": year2, # least cost for a period
        "park_name": park_name,
        "image_uri": image_uri,
        "explanation_requested": explanation_requested,  # used by planner
    }

    # ----- LOGGING -----
    print(f"[NLU] Final intent: {intent}")
    print(f"[NLU] Domain: {domain}")
    print(f"[NLU] Extracted slots: {slots}")

    return NLUResult(
        intent=intent,
        confidence=round(confidence, 3),
        slots=slots,
        raw_query=query
    )