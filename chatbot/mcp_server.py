#!/usr/bin/env python3
"""
Basin AI Insights — MCP Server
================================
FastMCP server exposing 25 tools and 6 prompts for petroleum basin analysis.

Architecture:
    User Query → LLM Orchestrator → MCP Tools → MeiliSearch (fuzzy) + PostgreSQL (precise) → Response

Tables: 15 parent tables + 27 child tables across 14 petroleum domains.
Basins: 35 basins/fields with deterministic UUID5 identifiers.

Transport: SSE on port 8000
"""

import os
import json
import logging
import psycopg2
import psycopg2.extras
import meilisearch
from fastmcp import FastMCP
from psycopg2 import pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("basin_ai")

# ── Config ──────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB", "alpha"),
    "user":     os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "secret"),
}
MEILI_URL   = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_KEY   = os.getenv("MEILI_API_KEY", "masterkey")
MEILI_INDEX = os.getenv("MEILI_INDEX", "basins-alpha")

# ── MeiliSearch client (official SDK) ───────────────────────────────────
meili = meilisearch.Client(MEILI_URL, MEILI_KEY)

mcp = FastMCP(
    name="basin-ai",
    instructions="""You are a petroleum basin intelligence assistant with access to 35 basins
across 14 geological domains (source rock, thermal history, reservoir quality, reservoir geometry,
structural style, seal properties, fluid properties, trap geometry, field reserves, production
recovery, alteration risk, formation water, migration, reservoir conditions).

All tools accept basin names directly — fuzzy matching handles typos and abbreviations.
Always state data confidence and flag missing values in your answers.
Never invent data — only reason over what tools return.
Always cite the data_source field from tool results verbatim.""",
)


# ── Database helpers ────────────────────────────────────────────────────

# def get_pg_connection():
#     return psycopg2.connect(**DB_CONFIG)
_pool = pool.ThreadedConnectionPool(1, 25, **DB_CONFIG)

def get_pg_connection():
    return _pool.getconn()


def query_one(sql, params=None):
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return dict(row) if row else {}
    finally:
        _pool.putconn(conn)   # ← was conn.close()

def query_all(sql, params=None):
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or ())
        return [dict(r) for r in cur.fetchall()]
    finally:
        _pool.putconn(conn)   # ← was conn.close()


def query_child_values(table, basin_id):
    """Fetch all values from a child table for a basin. Returns list of strings."""
    rows = query_all(f"SELECT value FROM {table} WHERE basin_id = %s ORDER BY value", (basin_id,))
    return [r["value"] for r in rows]


def clean_result(d):
    """Convert Decimal, None-heavy dicts to clean JSON-safe format. Remove internal IDs."""
    if not d:
        return d
    cleaned = {}
    for k, v in d.items():
        if k in ("id",):
            continue
        if hasattr(v, "as_integer_ratio"):  # Decimal → float
            cleaned[k] = float(v) if v is not None else None
        else:
            cleaned[k] = v
    return cleaned


# ── MeiliSearch helpers ─────────────────────────────────────────────────

def meili_search(query, limit=5, filters=None):
    """Search MeiliSearch basins index using official SDK. Returns list of hits."""
    try:
        params = {"limit": limit}
        if filters:
            params["filter"] = filters
        result = meili.index(MEILI_INDEX).search(query, params)
        return result.get("hits", [])
    except Exception as e:
        logger.warning(f"MeiliSearch error: {e}")
    return []


def resolve_basin_id(basin_name: str):
    """Resolve a basin name (fuzzy) to (basin_id, basin_name) via MeiliSearch.
    Falls back to PostgreSQL ILIKE if MeiliSearch is unavailable.
    """
    # Try MeiliSearch first
    hits = meili_search(basin_name, limit=1)
    if hits:
        return hits[0]["id"], hits[0]["basin_name"]

    # Fallback: PostgreSQL ILIKE
    row = query_one(
        "SELECT id, basin_name FROM basins WHERE basin_name ILIKE %s LIMIT 1",
        (f"%{basin_name}%",)
    )
    if row:
        return str(row["id"]), row["basin_name"]

    return None, None


# =====================================================================
# @tool: UTILITY (3)
# =====================================================================

@mcp.tool(
    description=(
        "Resolve a basin or field name to its unique identifier using fuzzy matching. "
        "Use this as the FIRST step when any tool needs a basin_name — it handles "
        "typos, abbreviations, and partial names. For example, 'volve' resolves to "
        "'Volve Field (Viking Graben)', 'barents' resolves to 'Norwegian Barents Sea', "
        "and 'CNS' resolves to 'Central North Sea (Central Graben)'. "
        "Returns the basin_id (UUID) and the canonical basin_name."
    )
)
def resolve_basin(basin_name: str) -> dict:
    """Resolve a fuzzy basin name to its canonical ID and name."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"No basin found matching '{basin_name}'. Use get_all_basins to see available basins."}
    return {"basin_id": bid, "basin_name": bname}


@mcp.tool(
    description=(
        "List all 35 basins and fields in the dataset with their names and index numbers. "
        "Use this when the user asks an open-ended question without specifying a basin, "
        "or when you need to iterate over all basins (e.g., 'compare maturity across all basins'). "
        "Returns basin_id, basin_name, and basin_index for each entry."
    )
)
def get_all_basins() -> list:
    """Return all basins in the dataset."""
    return query_all("SELECT id as basin_id, basin_name, basin_index FROM basins ORDER BY basin_index")


@mcp.tool(
    description=(
        "Search for basins matching a text query and optional filters using MeiliSearch. "
        "Supports fuzzy text matching across ALL indexed fields: kerogen types, depositional "
        "environments, lithologies, structural styles, trap types, seal lithologies, HC phases, "
        "field names, and more. Also supports faceted filtering on numeric midpoints. "
        "\n\nExamples:\n"
        "- search_basins(query='Type II turbidite') → basins with Type II kerogen AND turbidite reservoirs\n"
        "- search_basins(query='salt tectonics') → basins with salt-related structural styles\n"
        "- search_basins(query='evaporite seal') → basins with evaporite seal lithology\n"
        "- search_basins(query='', filters='porosity_avg_mid > 20') → basins with high porosity\n"
        "- search_basins(query='Oil', filters='hc_phases = Oil AND toc_mid > 5') → oil basins with rich source rocks\n"
        "\nUse this for screening questions like 'which basins have X?' or 'find basins similar to Y'."
    )
)
def search_basins(query: str, filters: str = "") -> list:
    """Search basins by text query with optional MeiliSearch filters."""
    hits = meili_search(query, limit=10, filters=filters if filters else None)
    results = []
    for h in hits:
        results.append({
            "basin_id": h.get("id"),
            "basin_name": h.get("basin_name"),
            "kerogen_types": h.get("kerogen_types"),
            "hc_phases": h.get("hc_phases"),
            "depositional_envs": h.get("depositional_envs"),
            "structural_styles": h.get("structural_styles"),
            "porosity_avg_mid": h.get("porosity_avg_mid"),
            "maturity_level": h.get("maturity_level"),
        })
    if not results:
        return [{"message": f"No basins found for query='{query}' filters='{filters}'. Try broader terms."}]
    return results


# =====================================================================
# @tool: SOURCE ROCK (2)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve complete source rock data for a specific basin. Returns kerogen types, "
        "TOC range (min/max/mid in wt%), Hydrogen Index range, vitrinite reflectance (Ro) range, "
        "maturity level, source rock age (period and Ma range), formation names, "
        "thickness range, organic facies classification, and data source citation. "
        "\n\nUse this for questions about source rock quality, organic richness, maturity, "
        "kerogen type, or when building a basin profile. The kerogen_types, formations, "
        "and organic_facies fields are arrays (a basin can have multiple source rocks)."
    )
)
def get_source_rock(basin_name: str) -> dict:
    """Get source rock properties for a basin including kerogen types, TOC, Ro, maturity."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}

    row = query_one("SELECT * FROM source_rock WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["kerogen_types"] = query_child_values("source_rock_kerogen_types", bid)
    result["formations"] = query_child_values("source_rock_formations", bid)
    result["organic_facies"] = query_child_values("source_rock_organic_facies", bid)
    return result


@mcp.tool(
    description=(
        "Compare source rock characteristics between two basins side-by-side. "
        "Returns both basin records with kerogen types, TOC ranges, Ro ranges, "
        "age ranges, and organic facies — plus computed overlap flags indicating "
        "whether TOC ranges overlap, Ro ranges overlap, and age ranges overlap. "
        "\n\nUse this when the user asks 'are the source rocks comparable between A and B?', "
        "'are there analogous source rock intervals?', or 'compare kerogen and maturity'."
    )
)
def compare_source_rocks(basin_a: str, basin_b: str) -> dict:
    """Compare source rock properties between two basins with overlap analysis."""
    a = get_source_rock(basin_a)
    b = get_source_rock(basin_b)
    if "error" in a or "error" in b:
        return {"error": f"Could not resolve one or both basins.", "basin_a": a, "basin_b": b}

    def ranges_overlap(a_min, a_max, b_min, b_max):
        if any(v is None for v in [a_min, a_max, b_min, b_max]):
            return None
        return a_max >= b_min and a_min <= b_max

    overlaps = {
        "toc_overlap": ranges_overlap(a.get("toc_min"), a.get("toc_max"), b.get("toc_min"), b.get("toc_max")),
        "ro_overlap": ranges_overlap(a.get("ro_min"), a.get("ro_max"), b.get("ro_min"), b.get("ro_max")),
        "hi_overlap": ranges_overlap(a.get("hi_min"), a.get("hi_max"), b.get("hi_min"), b.get("hi_max")),
        "age_overlap": ranges_overlap(a.get("age_ma_min"), a.get("age_ma_max"), b.get("age_ma_min"), b.get("age_ma_max")),
        "kerogen_match": bool(set(a.get("kerogen_types", [])) & set(b.get("kerogen_types", []))),
    }
    return {"basin_a": a, "basin_b": b, "overlap_analysis": overlaps}


# =====================================================================
# @tool: THERMAL HISTORY (2)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve thermal history data for a specific basin. Returns heat flow range "
        "(mW/m², min/max/mid), geothermal gradient range (°C/km), maximum burial depth range (m), "
        "paleo-surface temperature (°C), uplift/erosion estimate range (m), "
        "time of maximum burial (geological period), thermal event notes, and data source. "
        "\n\nUse this for questions about thermal maturation timing, heat flow comparison, "
        "whether a basin experienced uplift (which arrests maturation), or burial history."
    )
)
def get_thermal_history(basin_name: str) -> dict:
    """Get thermal history for a basin including heat flow, gradient, burial depth, uplift."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM thermal_history WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    return result


@mcp.tool(
    description=(
        "Compare thermal history between two basins side-by-side. Returns both records "
        "with heat flow, gradient, burial depth, uplift, and timing — plus computed "
        "differences (delta heat flow, delta gradient). "
        "\n\nUse this when the user asks 'how does the thermal history compare to the analogue?', "
        "'which basin has higher heat flow?', or 'has either basin been uplifted?'."
    )
)
def compare_thermal_history(basin_a: str, basin_b: str) -> dict:
    """Compare thermal history between two basins with computed differences."""
    a = get_thermal_history(basin_a)
    b = get_thermal_history(basin_b)
    if "error" in a or "error" in b:
        return {"error": "Could not resolve one or both basins.", "basin_a": a, "basin_b": b}

    def safe_diff(va, vb):
        if va is not None and vb is not None:
            return round(va - vb, 2)
        return None

    deltas = {
        "delta_heat_flow_mid": safe_diff(a.get("heat_flow_mid"), b.get("heat_flow_mid")),
        "delta_gradient_mid": safe_diff(a.get("geothermal_gradient_mid"), b.get("geothermal_gradient_mid")),
        "delta_max_burial_mid": safe_diff(a.get("max_burial_depth_mid_m"), b.get("max_burial_depth_mid_m")),
    }
    return {"basin_a": a, "basin_b": b, "deltas": deltas}


# =====================================================================
# @tool: RESERVOIR QUALITY + GEOMETRY (2)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve reservoir quality data for a specific basin. Returns porosity range "
        "(average and full range, min/max/mid in %), permeability range (mD), "
        "net-to-gross ratio range, depth range (m), primary reservoir formations "
        "(array), depositional environments (array), and lithologies (array). "
        "\n\nUse this for questions about reservoir quality, porosity-depth trends, "
        "NTG values, depositional environments, or when comparing reservoir properties "
        "between basins. NOTE: Data is at basin level (one range per basin), not per-play."
    )
)
def get_reservoir_quality(basin_name: str) -> dict:
    """Get reservoir quality for a basin: porosity, permeability, NTG, depth, lithology."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM reservoir_quality WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["primary_reservoirs"] = query_child_values("reservoir_primary_reservoirs", bid)
    result["depositional_envs"] = query_child_values("reservoir_depositional_envs", bid)
    result["lithologies"] = query_child_values("reservoir_lithologies", bid)
    return result


@mcp.tool(
    description=(
        "Retrieve reservoir geometry and architecture data for a specific basin. Returns "
        "geometry types (array — e.g., 'Sheet sand', 'Channel fill', 'Fan lobe'), "
        "continuity description, sand body thickness range (m), sand body width range (m), "
        "stacking patterns (array — e.g., 'Progradational', 'Aggradational'), "
        "and facies associations (array — e.g., 'Distributary channel', 'Mouth bar'). "
        "\n\nUse this for questions about depositional facies geometry, sand body dimensions, "
        "reservoir connectivity, or stacking architecture. Complements get_reservoir_quality "
        "which provides porosity/perm/NTG but not geometry details."
    )
)
def get_reservoir_geometry(basin_name: str) -> dict:
    """Get reservoir geometry: sand body dimensions, stacking, facies associations."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM reservoir_geometry WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["geometry_types"] = query_child_values("reservoir_geom_geometry_types", bid)
    result["stacking_patterns"] = query_child_values("reservoir_geom_stacking_patterns", bid)
    result["facies_associations"] = query_child_values("reservoir_geom_facies_assocs", bid)
    return result


# =====================================================================
# @tool: STRUCTURAL STYLE (2)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve structural geology data for a specific basin. Returns basin types "
        "(array — e.g., 'Failed rift', 'Foreland'), tectonic settings (array), "
        "structural styles (array — e.g., 'Extensional', 'Compressional', 'Salt-related'), "
        "deformation mechanisms (array — e.g., 'Rifting', 'Salt tectonics', 'Tectonic inversion'), "
        "dominant trap types (array — e.g., 'Tilted fault block', 'Drape', 'Inversion anticline'), "
        "fault types (array — e.g., 'Normal', 'Reverse', 'Strike-slip'), "
        "structural complexity rating, and data source. "
        "\n\nUse this for questions about structural style, deformation mechanisms, "
        "trap types, or whether two basins share similar tectonic settings."
    )
)
def get_structural_style(basin_name: str) -> dict:
    """Get structural geology: basin type, tectonics, styles, traps, faults, complexity."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM structural_style WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["basin_types"] = query_child_values("structural_basin_types", bid)
    result["tectonic_settings"] = query_child_values("structural_tectonic_settings", bid)
    result["structural_styles"] = query_child_values("structural_styles_mv", bid)
    result["deformation_mechanisms"] = query_child_values("structural_deformation_mechanisms", bid)
    result["trap_types"] = query_child_values("structural_trap_types", bid)
    result["fault_types"] = query_child_values("structural_fault_types", bid)
    return result


@mcp.tool(
    description=(
        "Compare structural styles between two basins. Returns both records side-by-side "
        "with match flags indicating: same structural style? same deformation mechanism? "
        "same trap types? same fault types? same basin type? "
        "\n\nUse when the user asks 'is the structural style similar?', "
        "'are the trap types compatible?', or 'is the deformation mechanism analogous?'."
    )
)
def compare_structural_styles(basin_a: str, basin_b: str) -> dict:
    """Compare structural styles between two basins with match flags."""
    a = get_structural_style(basin_a)
    b = get_structural_style(basin_b)
    if "error" in a or "error" in b:
        return {"error": "Could not resolve one or both basins.", "basin_a": a, "basin_b": b}

    def array_overlap(la, lb):
        sa, sb = set(la or []), set(lb or [])
        return {"shared": list(sa & sb), "only_a": list(sa - sb), "only_b": list(sb - sa), "match": bool(sa & sb)}

    matches = {
        "structural_styles": array_overlap(a.get("structural_styles"), b.get("structural_styles")),
        "deformation_mechanisms": array_overlap(a.get("deformation_mechanisms"), b.get("deformation_mechanisms")),
        "trap_types": array_overlap(a.get("trap_types"), b.get("trap_types")),
        "fault_types": array_overlap(a.get("fault_types"), b.get("fault_types")),
        "basin_types": array_overlap(a.get("basin_types"), b.get("basin_types")),
        "complexity_match": a.get("structural_complexity") == b.get("structural_complexity"),
    }
    return {"basin_a": a, "basin_b": b, "match_analysis": matches}


# =====================================================================
# @tool: SEAL (1 consolidated)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve ALL seal data for a specific basin in a single call. Returns "
        "seal lithologies (array — e.g., 'Shale', 'Evaporite', 'Marl'), "
        "seal formations (array), thickness range (m, min/max/mid), "
        "lateral extent (text — 'Basin-wide', 'Regional', 'Local'), "
        "continuity (text — 'Continuous', 'Discontinuous'), "
        "capillary pressure range (psi, min/max/mid — NOTE: mostly estimated, not lab-measured), "
        "tested_by_accumulation (boolean — whether known HC accumulations prove the seal works), "
        "tested_by_accumulation_desc (details — e.g., 'numerous giant fields', 'Statfjord, Gullfaks'), "
        "and data source citation. "
        "\n\nThis SINGLE tool replaces what was previously three separate seal tools. "
        "Use for ANY question about seal quality, capillary pressure, seal lithology, "
        "lateral extent, or whether the seal has been proven by discoveries."
    )
)
def get_seal_data(basin_name: str) -> dict:
    """Get all seal properties: lithology, thickness, extent, Pc, tested status."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM seal_properties WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["seal_lithologies"] = query_child_values("seal_lithologies", bid)
    result["seal_formations"] = query_child_values("seal_formations", bid)
    return result


# =====================================================================
# @tool: FLUID PROPERTIES + HC CONSISTENCY (2)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve fluid properties for a specific basin. Returns API gravity range "
        "(°, min/max/mid), GOR range (scf/bbl), bubble point pressure range (psi), "
        "condensate yield range (bbl/MMscf — often NULL for oil basins), "
        "oil viscosity range (cp), pour point range (°C), HC phases (array — e.g., "
        "['Oil', 'Gas'] or ['Gas (dry)'] or ['Oil', 'Condensate']), and data source. "
        "\n\nNOTE: ~8 gas-dominant basins have NULL API gravity — this is correct, not a data gap. "
        "Bubble point is ~30% NULL, condensate yield ~80% NULL. "
        "\n\nUse for questions about API gravity, GOR, fluid types, or when comparing "
        "oil properties between basins."
    )
)
def get_fluid_properties(basin_name: str) -> dict:
    """Get fluid properties: API, GOR, bubble point, viscosity, HC phases."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM fluid_properties WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["hc_phases"] = query_child_values("fluid_hc_phases", bid)
    return result


@mcp.tool(
    description=(
        "Assess whether the recorded HC phase is consistent with the source rock's "
        "kerogen type and thermal maturity. Cross-references source_rock (kerogen type, Ro) "
        "with fluid_properties (HC phases) using petroleum geochemistry rules: "
        "\n- Type I/II kerogen + Ro 0.5-1.0% → expect OIL"
        "\n- Type II kerogen + Ro 1.0-1.3% → expect OIL & GAS"
        "\n- Type III kerogen + any maturity → expect GAS"
        "\n- Type II kerogen + Ro > 1.3% → expect DRY GAS"
        "\n\nReturns the kerogen type, Ro, recorded HC phases, expected phase, "
        "and a consistency flag (CONSISTENT / INCONSISTENT) with explanation. "
        "\n\nUse for question 'Is the HC phase consistent with thermal maturity and source facies?'."
    )
)
def assess_hc_consistency(basin_name: str) -> dict:
    """Cross-check kerogen type × maturity vs recorded HC phase."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}

    sr = query_one("SELECT ro_min, ro_max, ro_mid, maturity_level FROM source_rock WHERE basin_id = %s", (bid,))
    kerogen_types = query_child_values("source_rock_kerogen_types", bid)
    hc_phases = query_child_values("fluid_hc_phases", bid)

    ro_mid = sr.get("ro_mid")
    maturity = sr.get("maturity_level", "")
    kt_str = ", ".join(kerogen_types) if kerogen_types else "Unknown"

    # Determine expected phase
    expected = "Unknown"
    if any("III" in k for k in kerogen_types):
        expected = "Gas"
    elif any("II" in k for k in kerogen_types) or any("I" in k and "II" not in k for k in kerogen_types):
        if ro_mid is not None:
            if ro_mid < 0.5:
                expected = "Immature (no generation expected)"
            elif ro_mid <= 1.0:
                expected = "Oil"
            elif ro_mid <= 1.3:
                expected = "Oil & Gas"
            else:
                expected = "Dry Gas"
        else:
            expected = "Oil (assumed from kerogen type)"

    hc_str = ", ".join(hc_phases) if hc_phases else "Unknown"
    # Check consistency
    consistent = True
    explanation = "Recorded HC phase matches expected phase from kerogen type and maturity."
    if expected == "Unknown" or hc_str == "Unknown":
        consistent = None
        explanation = "Insufficient data to assess consistency."
    elif "Oil" in expected and "Oil" not in hc_str and "oil" not in hc_str.lower():
        consistent = False
        explanation = f"Expected '{expected}' from {kt_str} at Ro={ro_mid}%, but recorded phase is '{hc_str}'."
    elif "Gas" in expected and "Gas" not in hc_str and "gas" not in hc_str.lower():
        consistent = False
        explanation = f"Expected '{expected}' from {kt_str} at Ro={ro_mid}%, but recorded phase is '{hc_str}'."

    return {
        "basin_name": bname,
        "kerogen_types": kerogen_types,
        "ro_min": float(sr.get("ro_min")) if sr.get("ro_min") else None,
        "ro_max": float(sr.get("ro_max")) if sr.get("ro_max") else None,
        "ro_mid": float(ro_mid) if ro_mid else None,
        "maturity_level": maturity,
        "recorded_hc_phases": hc_phases,
        "expected_phase": expected,
        "consistent": consistent,
        "explanation": explanation,
        "data_source": sr.get("data_source", ""),
    }


# =====================================================================
# @tool: FIELD RESERVES + PRODUCTION + CONDITIONS (3)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve field-level reserves and exploration data for a specific basin. Returns "
        "key fields (array of field names), total basin reserves (value + description), "
        "largest field reserves (value + description), first major discovery year, "
        "field count, drive mechanisms (array), recovery factor range (%, min/max/mid), "
        "and data source. "
        "\n\nUse for questions about exploration history, field sizes, discoveries, "
        "recovery factors, or drive mechanisms. NOTE: reserve figures are basin-level "
        "aggregates — we do not have per-field reserve breakdowns for distribution fitting."
    )
)
def get_field_reserves(basin_name: str) -> dict:
    """Get field reserves, discovery history, drive mechanisms, recovery factors."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM field_reserves WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["key_fields"] = query_child_values("field_key_fields", bid)
    result["drive_mechanisms"] = query_child_values("field_drive_mechanisms", bid)
    return result


@mcp.tool(
    description=(
        "Retrieve production and recovery data for a specific basin. Returns "
        "recovery factor range (%, min/max/mid + description), OOIP (MMbbl, value + description), "
        "cumulative oil production (MMbbl, value + description), water cut description, "
        "drive mechanisms (array), and IOR/EOR methods (array — e.g., 'Water injection', "
        "'WAG', 'Polymer flood'). "
        "\n\nUse for questions about recovery factors, drive mechanisms, "
        "production history, or enhanced recovery methods."
    )
)
def get_production_recovery(basin_name: str) -> dict:
    """Get production/recovery data: RF, OOIP, cumulative oil, drive mechanisms, IOR/EOR."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM production_recovery WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["drive_mechanisms"] = query_child_values("production_drive_mechanisms", bid)
    result["ior_eor_methods"] = query_child_values("production_ior_eor_methods", bid)
    return result


@mcp.tool(
    description=(
        "Retrieve reservoir conditions data for a specific basin. Returns initial "
        "pressure range (psi), pressure gradient range (psi/ft), overpressure ratio range, "
        "reservoir temperature range (°C), OWC depth range (m), GOC depth range (m), "
        "GOC description (for gas reservoirs where GOC replaces OWC), "
        "aquifer strength description, and data source. "
        "\n\nUse for questions about pressure regimes, overpressure, fluid contacts, "
        "or aquifer support. Compare two basins to assess if they operate under similar conditions."
    )
)
def get_reservoir_conditions(basin_name: str) -> dict:
    """Get reservoir conditions: pressure, temperature, fluid contacts, aquifer strength."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM reservoir_conditions WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    return result


# =====================================================================
# @tool: ALTERATION, TRAP, FORMATION WATER, MIGRATION (4)
# =====================================================================

@mcp.tool(
    description=(
        "Retrieve alteration risk data for a specific basin. Returns biodegradation risk "
        "level (text — 'Low', 'Moderate', 'High'), biodegradation level range (Peters & "
        "Moldowan scale), reservoir temperature range (°C — key: >80°C means biodeg unlikely), "
        "H2S concentration range (%), gas washing assessment, water leg presence (boolean + "
        "description), alteration notes, and data source. "
        "\n\nUse for questions about biodegradation risk, H2S souring, gas washing, "
        "or reservoir alteration. The 80°C temperature threshold is the primary "
        "biodegradation risk discriminator."
    )
)
def get_alteration_risk(basin_name: str) -> dict:
    """Get alteration risk: biodegradation, H2S, gas washing, reservoir temperature."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM alteration_risk WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    return result


@mcp.tool(
    description=(
        "Retrieve trap geometry data for a specific basin. Returns trap types (array), "
        "closure area range (km²), spill point depth range (m), crest depth range (m), "
        "maximum column height range (m), fill fraction range (0-1), and data source. "
        "\n\nUse for questions about spill point geometry, column heights, "
        "fill-to-spill examples, trap size, or closure area. "
        "Fill fraction near 1.0 = filled to spill point = seal proven to that column height."
    )
)
def get_trap_geometry(basin_name: str) -> dict:
    """Get trap geometry: closure area, spill point, column height, fill fraction."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM trap_geometry WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["trap_types"] = query_child_values("trap_geom_trap_types", bid)
    return result


@mcp.tool(
    description=(
        "Retrieve formation water data for a specific basin. Returns salinity range "
        "(ppm TDS, min/max/mid), water types (array — e.g., 'NaCl', 'CaCl2'), "
        "major ion concentrations (Na, Ca, Cl, SO4 in ppm), resistivity of formation "
        "water (Rw range in ohm.m), and data source. "
        "\n\nUse for questions about formation water salinity, water chemistry, "
        "or Rw values needed for Sw (water saturation) calculations in log interpretation."
    )
)
def get_formation_water(basin_name: str) -> dict:
    """Get formation water: salinity, water type, ion concentrations, Rw."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM formation_water WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["water_types"] = query_child_values("formation_water_types", bid)
    return result


@mcp.tool(
    description=(
        "Retrieve migration pathway data for a specific basin. Returns carrier beds "
        "(array — e.g., 'Porous sandstone', 'Faulted carrier'), migration distance range (km), "
        "migration direction description, migration mechanisms (array — e.g., 'Buoyancy', "
        "'Compaction expulsion'), kitchen locations (array), kitchen depth range (m), "
        "vertical migration flag, and data source. "
        "\n\nUse for questions about migration pathways, migration distances, "
        "carrier bed identification, or kitchen-to-trap relationships."
    )
)
def get_migration(basin_name: str) -> dict:
    """Get migration data: carrier beds, distance, direction, mechanisms, kitchen info."""
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}
    row = query_one("SELECT * FROM migration WHERE basin_id = %s", (bid,))
    result = clean_result(row)
    result["basin_name"] = bname
    result["carrier_beds"] = query_child_values("migration_carrier_beds", bid)
    result["kitchen_locations"] = query_child_values("migration_kitchen_locations", bid)
    result["migration_mechanisms"] = query_child_values("migration_mechanisms_mv", bid)
    return result


# =====================================================================
# @tool: RANKING / AGGREGATION (3)
# =====================================================================

RANK_METRIC_MAP = {
    "heat_flow": ("thermal_history", "heat_flow_mid"),
    "geothermal_gradient": ("thermal_history", "geothermal_gradient_mid"),
    "porosity": ("reservoir_quality", "porosity_avg_mid"),
    "permeability": ("reservoir_quality", "perm_avg_mid"),
    "ntg": ("reservoir_quality", "ntg_mid"),
    "api_gravity": ("fluid_properties", "api_gravity_mid"),
    "gor": ("fluid_properties", "gor_mid"),
    "recovery_factor": ("field_reserves", "recovery_factor_mid"),
    "ro": ("source_rock", "ro_mid"),
    "toc": ("source_rock", "toc_mid"),
    "hi": ("source_rock", "hi_mid"),
    "seal_thickness": ("seal_properties", "seal_thickness_mid_m"),
    "column_height": ("trap_geometry", "max_column_height_mid_m"),
    "reservoir_temp": ("alteration_risk", "reservoir_temp_mid_c"),
}


@mcp.tool(
    description=(
        "Rank all basins by a specified numeric metric, returning the top N. "
        f"Available metrics: {', '.join(RANK_METRIC_MAP.keys())}. "
        "\n\nExamples:\n"
        "- rank_basins(metric='heat_flow', top_n=5) → 5 basins with highest heat flow\n"
        "- rank_basins(metric='porosity', top_n=10) → top 10 by average porosity\n"
        "- rank_basins(metric='recovery_factor') → basins ranked by recovery factor\n"
        "\n\nUse for questions like 'which basins have the highest X?', "
        "'rank basins by porosity', or 'which basin is most prospective by metric Y?'."
    )
)
def rank_basins(metric: str, top_n: int = 10) -> list:
    """Rank basins by a numeric metric. Use metric names like 'porosity', 'heat_flow', 'toc'."""
    if metric not in RANK_METRIC_MAP:
        return [{"error": f"Unknown metric '{metric}'. Available: {', '.join(RANK_METRIC_MAP.keys())}"}]

    table, column = RANK_METRIC_MAP[metric]
    sql = f"""
        SELECT b.basin_name, t.{column} as value, t.data_source
        FROM {table} t JOIN basins b ON t.basin_id = b.id
        WHERE t.{column} IS NOT NULL
        ORDER BY t.{column} DESC
        LIMIT %s
    """
    rows = query_all(sql, (top_n,))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["metric"] = metric
    return rows


@mcp.tool(
    description=(
        "Get the distribution of thermal maturity levels across all basins. "
        "Returns each maturity level (e.g., 'Oil Window', 'Gas Window', 'Immature'), "
        "the count of basins in that category, and the list of basin names. "
        "\n\nUse for questions like 'what is the maturity distribution?', "
        "'how many basins are in the oil window?', or 'which basins are immature?'."
    )
)
def get_maturity_distribution() -> list:
    """Get distribution of maturity levels across all basins."""
    return query_all("""
        SELECT sr.maturity_level, COUNT(*) as basin_count,
               STRING_AGG(b.basin_name, ', ' ORDER BY b.basin_index) as basins
        FROM source_rock sr
        JOIN basins b ON sr.basin_id = b.id
        WHERE sr.maturity_level IS NOT NULL
        GROUP BY sr.maturity_level
        ORDER BY basin_count DESC
    """)


@mcp.tool(
    description=(
        "Get a summary of depositional environments across all basins. "
        "Returns each depositional environment type, the count of basins where it occurs, "
        "and the list of basin names. "
        "\n\nUse for questions like 'which depositional environments are represented?', "
        "'how many basins have turbidite reservoirs?', or 'list all deltaic basins'."
    )
)
def get_depositional_env_summary() -> list:
    """Get distribution of depositional environments across all basins."""
    return query_all("""
        SELECT rde.value as depositional_env, COUNT(DISTINCT b.basin_name) as basin_count,
               STRING_AGG(DISTINCT b.basin_name, ', ' ORDER BY b.basin_name) as basins
        FROM reservoir_depositional_envs rde
        JOIN basins b ON rde.basin_id = b.id
        GROUP BY rde.value
        ORDER BY basin_count DESC
    """)


@mcp.tool(
    description=(
        "Find the best petroleum basin analogues for a target basin using multi-dimensional "
        "scoring across 6 axes (13 points total): "
        "\n1. Source Rock (0-3 pts): kerogen type match + TOC overlap + age overlap"
        "\n2. Thermal Maturity (0-2 pts): Ro range overlap + maturity level match"
        "\n3. Reservoir Quality (0-3 pts): lithology match + porosity overlap + depositional env match"
        "\n4. Structural Style (0-2 pts): structural style match + deformation mechanism match"
        "\n5. Seal Quality (0-2 pts): seal lithology match + tested-by-accumulation match"
        "\n6. HC Phase (0-1 pt): HC phase match"
        "\n\nReturns the top 5 analogues ranked by total score with per-axis breakdown. "
        "Score >= 9: Strong analogue. 6-8: Moderate. 3-5: Weak. <3: Poor. "
        "\n\nUse for questions like 'find analogues for Basin X', "
        "'which basin is most similar to the Lusitanian Basin?', or "
        "'rank basins by analogue similarity'."
    )
)
def find_best_analogue(basin_name: str) -> dict:
    """Find best analogue basins using 6-axis, 13-point scoring algorithm.
    Uses bulk pre-fetch to avoid N×10 query pattern inside scoring loop.
    """
    bid, bname = resolve_basin_id(basin_name)
    if not bid:
        return {"error": f"Basin '{basin_name}' not found."}

    # ------------------------------------------------------------------
    # STEP 1: Bulk pre-fetch all data in 12 queries (replaces 350 queries)
    # ------------------------------------------------------------------

    # Parent tables: basin_id (UUID) -> row dict
    all_sr   = {str(r["basin_id"]): r for r in query_all("SELECT * FROM source_rock")}
    all_rq   = {str(r["basin_id"]): r for r in query_all("SELECT * FROM reservoir_quality")}
    all_seal = {str(r["basin_id"]): r for r in query_all(
                    "SELECT basin_id, tested_by_accumulation FROM seal_properties")}

    # Child tables: basin_id (str) -> set of values
    def bulk_child(table):
        rows = query_all(f"SELECT basin_id, value FROM {table}")
        out = {}
        for r in rows:
            key = str(r["basin_id"])
            out.setdefault(key, set()).add(r["value"])
        return out

    all_kt     = bulk_child("source_rock_kerogen_types")
    all_lith   = bulk_child("reservoir_lithologies")
    all_dep    = bulk_child("reservoir_depositional_envs")
    all_ss     = bulk_child("structural_styles_mv")
    all_mech   = bulk_child("structural_deformation_mechanisms")
    all_sl     = bulk_child("seal_lithologies")
    all_hc     = bulk_child("fluid_hc_phases")

    # ------------------------------------------------------------------
    # STEP 2: Extract target basin data from pre-fetched dicts
    # ------------------------------------------------------------------

    target_sr        = all_sr.get(bid, {})
    target_kt        = all_kt.get(bid, set())
    target_rq        = all_rq.get(bid, {})
    target_lith      = all_lith.get(bid, set())
    target_dep       = all_dep.get(bid, set())
    target_ss        = all_ss.get(bid, set())
    target_mech      = all_mech.get(bid, set())
    target_seal_lith = all_sl.get(bid, set())
    target_seal      = all_seal.get(bid, {})
    target_hc        = all_hc.get(bid, set())

    # ------------------------------------------------------------------
    # STEP 3: Scoring loop — pure in-memory, zero DB calls
    # ------------------------------------------------------------------

    def overlap(a_min, a_max, b_min, b_max):
        if any(v is None for v in [a_min, a_max, b_min, b_max]):
            return False
        return float(a_max) >= float(b_min) and float(a_min) <= float(b_max)

    all_basins = query_all(
        "SELECT id, basin_name FROM basins WHERE id != %s ORDER BY basin_name", (bid,)
    )

    scores = []
    for other in all_basins:
        oid   = str(other["id"])
        oname = other["basin_name"]
        o_sr  = all_sr.get(oid, {})

        axis_scores = {}

        # 1. Source Rock (0-3)
        o_kt  = all_kt.get(oid, set())
        sr_score  = 0
        kt_match  = bool(target_kt & o_kt)
        toc_ovl   = overlap(target_sr.get("toc_min"),    target_sr.get("toc_max"),
                            o_sr.get("toc_min"),         o_sr.get("toc_max"))
        age_ovl   = overlap(target_sr.get("age_ma_min"), target_sr.get("age_ma_max"),
                            o_sr.get("age_ma_min"),      o_sr.get("age_ma_max"))
        if kt_match:
            sr_score = 1
            if toc_ovl: sr_score += 1
            if age_ovl: sr_score += 1
        axis_scores["source_rock"] = sr_score

        # 2. Thermal Maturity (0-2)
        ro_ovl    = overlap(target_sr.get("ro_min"), target_sr.get("ro_max"),
                            o_sr.get("ro_min"),      o_sr.get("ro_max"))
        mat_match = (
            (target_sr.get("maturity_level") or "").lower() ==
            (o_sr.get("maturity_level") or "").lower()
        )
        th_score = 0
        if ro_ovl: th_score += 1
        if mat_match and target_sr.get("maturity_level"): th_score += 1
        axis_scores["thermal_maturity"] = th_score

        # 3. Reservoir Quality (0-3)
        o_rq   = all_rq.get(oid, {})
        o_lith = all_lith.get(oid, set())
        o_dep  = all_dep.get(oid, set())
        rq_score = 0
        if target_lith & o_lith: rq_score += 1
        if overlap(target_rq.get("porosity_avg_min"), target_rq.get("porosity_avg_max"),
                   o_rq.get("porosity_avg_min"),      o_rq.get("porosity_avg_max")):
            rq_score += 1
        if target_dep & o_dep: rq_score += 1
        axis_scores["reservoir_quality"] = rq_score

        # 4. Structural Style (0-2)
        o_ss   = all_ss.get(oid, set())
        o_mech = all_mech.get(oid, set())
        ss_score = 0
        if target_ss   & o_ss:   ss_score += 1
        if target_mech & o_mech: ss_score += 1
        axis_scores["structural_style"] = ss_score

        # 5. Seal Quality (0-2)
        o_sl   = all_sl.get(oid, set())
        o_seal = all_seal.get(oid, {})
        seal_score = 0
        if target_seal_lith & o_sl: seal_score += 1
        if target_seal.get("tested_by_accumulation") and o_seal.get("tested_by_accumulation"):
            seal_score += 1
        axis_scores["seal_quality"] = seal_score

        # 6. HC Phase (0-1)
        o_hc = all_hc.get(oid, set())
        axis_scores["hc_phase"] = 1 if target_hc & o_hc else 0

        total = sum(axis_scores.values())
        scores.append({
            "basin_name": oname,
            "total_score": total,
            "max_score": 13,
            "axis_scores": axis_scores,
        })

    scores.sort(key=lambda x: x["total_score"], reverse=True)

    return {
        "target_basin": bname,
        "top_analogues": scores[:5],
        "scoring_method": (
            "6-axis (source_rock:3, thermal:2, reservoir:3, "
            "structural:2, seal:2, hc_phase:1) = 13 max"
        ),
    }


# =====================================================================
# @prompt: SYNTHESIS TEMPLATES (6)
# =====================================================================

@mcp.prompt(
    name="basin_profile",
    description=(
        "Generate a comprehensive petroleum geology profile for a basin. "
        "Instructs the LLM to call ALL domain tools for one basin and produce "
        "a structured narrative covering source rock, thermal history, reservoir, "
        "structure, seal, fluids, fields, production, and risks."
    )
)
def basin_profile(basin_name: str) -> str:
    return f"""You are a petroleum geologist preparing a basin screening report for {basin_name}.

Call these tools IN ORDER to gather all data:
1. get_source_rock('{basin_name}')
2. get_thermal_history('{basin_name}')
3. get_reservoir_quality('{basin_name}')
4. get_reservoir_geometry('{basin_name}')
5. get_structural_style('{basin_name}')
6. get_seal_data('{basin_name}')
7. get_fluid_properties('{basin_name}')
8. get_trap_geometry('{basin_name}')
9. get_field_reserves('{basin_name}')
10. get_production_recovery('{basin_name}')
11. get_alteration_risk('{basin_name}')
12. get_formation_water('{basin_name}')
13. get_migration('{basin_name}')
14. get_reservoir_conditions('{basin_name}')

Then synthesize into a structured profile with these sections:
1. Source Rock & Maturity — kerogen types, TOC, Ro, maturity level
2. Thermal History — heat flow, gradient, burial, uplift events
3. Reservoir Quality & Geometry — porosity, perm, NTG, depositional environment, sand body geometry
4. Structural Setting & Traps — basin type, deformation, trap types, complexity
5. Seal Quality — lithology, thickness, extent, Pc, field-tested status
6. Fluid Properties — API, GOR, HC phase
7. Trap Geometry — closure area, column height, fill fraction
8. Migration — carrier beds, distance, mechanisms
9. Exploration & Production — discoveries, reserves, RF, drive mechanisms, IOR/EOR
10. Reservoir Conditions — pressure, temperature, contacts, aquifer
11. Formation Water — salinity, water type
12. Alteration Risks — biodegradation, H2S, gas washing

IMPORTANT: Cite data_source from each tool result verbatim at the end of each section.
Present all ranges as "X-Y (mid: Z)" format. Do not invent data not returned by tools."""


@mcp.prompt(
    name="basin_comparison",
    description=(
        "Generate a structured comparison between two basins across all petroleum domains. "
        "Instructs the LLM to call comparison tools and domain tools for both basins, "
        "then produce a side-by-side analysis with match/mismatch assessment per domain."
    )
)
def basin_comparison(basin_a: str, basin_b: str) -> str:
    return f"""You are a petroleum geologist comparing {basin_a} and {basin_b} as potential analogues.

Call these tools:
1. compare_source_rocks('{basin_a}', '{basin_b}')
2. compare_thermal_history('{basin_a}', '{basin_b}')
3. compare_structural_styles('{basin_a}', '{basin_b}')
4. get_reservoir_quality('{basin_a}') and get_reservoir_quality('{basin_b}')
5. get_seal_data('{basin_a}') and get_seal_data('{basin_b}')
6. get_fluid_properties('{basin_a}') and get_fluid_properties('{basin_b}')

For EACH domain, state:
- What MATCHES (overlapping ranges, shared categories)
- What DIFFERS (non-overlapping ranges, different categories)
- Geological SIGNIFICANCE of differences

Conclude with:
- Overall analogue suitability assessment (Strong / Moderate / Weak / Poor)
- Which risk elements the analogue IS valid for
- Which risk elements require caution

Cite data_source from each tool result. Present ranges in full (min-max, mid)."""


@mcp.prompt(
    name="analogue_assessment",
    description=(
        "Interpret analogue scoring results for a basin. Instructs the LLM to call "
        "find_best_analogue and explain WHY each top analogue is a good or weak match, "
        "with geological reasoning per axis."
    )
)
def analogue_assessment(basin_name: str) -> str:
    return f"""You are a petroleum geologist assessing exploration analogues for {basin_name}.

Call find_best_analogue('{basin_name}') to get scored results.

For the top 3 analogues, explain:
1. WHY it scores highly — which axes match and what that means geologically
2. WHERE it diverges — which axes scored 0 and what risk that implies
3. SUITABILITY — is this analogue reliable for volumetric estimation? For risk assessment? For development planning?

Present as a ranked table:
| Rank | Basin | Total Score | Source Rock | Thermal | Reservoir | Structure | Seal | HC Phase |

Then provide a narrative paragraph for each of the top 3 analogues explaining the geological reasoning.

If no basin scores >= 7/13, state that no strong analogues exist in the current dataset and explain why."""


@mcp.prompt(
    name="hc_consistency_check",
    description=(
        "Assess HC phase consistency across one or all basins. Instructs the LLM to "
        "call assess_hc_consistency (per basin or in a loop) and present results as a "
        "summary table with explanations for any inconsistencies."
    )
)
def hc_consistency_check(scope: str = "all") -> str:
    if scope.lower() == "all":
        return """You are a petroleum geochemist performing a quality check across all basins.

Call get_all_basins() to get the full list.
Then call assess_hc_consistency() for EACH basin.

Present results as a table:
| Basin | Kerogen Type | Ro (mid) | Maturity | HC Phase | Expected | Status |

For any INCONSISTENT basins, provide a geological explanation:
- Could biodegradation have altered the oil to heavy oil?
- Could mixing from multiple source kitchens explain the mismatch?
- Is there a data quality issue?

Summary: State X of 35 basins are CONSISTENT, Y are INCONSISTENT."""
    else:
        return f"""You are a petroleum geochemist assessing HC phase consistency for {scope}.

Call assess_hc_consistency('{scope}').

Interpret the result:
- State the kerogen type, Ro, and maturity level
- State the expected HC phase based on petroleum geochemistry rules
- State whether the recorded phase matches
- If inconsistent, provide possible geological explanations

Rules reference:
- Type I/II + Ro 0.5-1.0% → Oil
- Type II + Ro 1.0-1.3% → Oil & Gas
- Type III → Gas-prone regardless of maturity
- Type II + Ro > 1.3% → Dry Gas / Condensate"""


@mcp.prompt(
    name="seal_risk_assessment",
    description=(
        "Perform a structured seal risk assessment for a basin. Instructs the LLM to "
        "call get_seal_data and rate seal quality across 5 dimensions with an overall risk rating."
    )
)
def seal_risk_assessment(basin_name: str) -> str:
    return f"""You are a seal integrity specialist assessing {basin_name}.

Call get_seal_data('{basin_name}').

Rate each dimension:
1. Seal Lithology: evaporite = Excellent, marine shale = Good, lacustrine shale = Fair, siltstone = Poor
2. Thickness: >200m = Excellent, 100-200m = Good, 50-100m = Fair, <50m = Poor
3. Lateral Extent: Basin-wide = Excellent, Regional = Good, Local = Fair
4. Capillary Pressure: >1000 psi = Excellent, 500-1000 = Good, 200-500 = Fair, <200 = Poor
5. Field-Tested: Yes = Proven (reduces risk), No = Unproven

NOTE: Capillary pressure values in this dataset are mostly ESTIMATED from lithology analogues,
not lab-measured MICP data. State this caveat in your assessment.

Provide overall seal risk: Low / Moderate / High / Critical
And a brief statement on implications for exploration and development."""


@mcp.prompt(
    name="biodegradation_assessment",
    description=(
        "Assess biodegradation and alteration risk for a basin. Instructs the LLM to "
        "call alteration risk and reservoir conditions tools, then apply temperature-based "
        "rules to rate risk with development implications."
    )
)
def biodegradation_assessment(basin_name: str) -> str:
    return f"""You are a petroleum alteration specialist assessing {basin_name}.

Call get_alteration_risk('{basin_name}') and get_reservoir_conditions('{basin_name}').

Apply these rules:
1. BIODEGRADATION:
   - Reservoir T > 80°C → Low risk (pasteurisation temperature exceeded)
   - Reservoir T 60-80°C → Moderate risk (borderline)
   - Reservoir T < 60°C → High risk (biodegradation likely active)
   
2. H2S / SOURING:
   - H2S > 3% → Severe (specialist materials + HSE protocols required)
   - H2S 0.5-3% → Moderate (sour service materials)
   - H2S < 0.5% → Low
   
3. GAS WASHING:
   - Evidence of gas washing → residual heavy oil risk, API may be misleadingly low

State:
- Biodegradation risk level with temperature evidence
- H2S risk level with concentration data
- Gas washing assessment
- Overall alteration risk: Low / Moderate / High
- Implications for development: material selection, facility design, production strategy"""


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    logger.info("Starting Basin AI Insights MCP Server")
    logger.info(f"PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    logger.info(f"MeiliSearch: {MEILI_URL} index={MEILI_INDEX}")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)