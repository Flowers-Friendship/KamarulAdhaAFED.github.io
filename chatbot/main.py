#!/usr/bin/env python3
"""
Basin AI Web Server
====================
FastAPI backend that wraps the MCP orchestrator and streams agentic workflow
events to the frontend via Server-Sent Events (SSE).

Events streamed to frontend:
    - status      : Connection/progress messages
    - iteration   : Agentic loop iteration start
    - tool_calls  : Batch of tool calls the LLM requested
    - tool_result : Individual tool execution result
    - thinking    : LLM reasoning content (if present alongside tool calls)
    - answer      : Final synthesised answer
    - error       : Error messages
    - done        : Stream complete

Usage:
    uvicorn web_server:app --host 0.0.0.0 --port 8080 --reload

Dependencies:
    pip install fastapi uvicorn sse-starlette mcp meilisearch psycopg2-binary httpx
"""

import os
import json
import asyncio
import logging
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from mcp import ClientSession
from mcp.client.sse import sse_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_server")

# ── Config ──────────────────────────────────────────────────────────────
MCP_SSE_URL = os.getenv("MCP_SSE_URL", "http://localhost:8000/sse")
LLM_URL     = os.getenv("LLM_URL", "https://magic.sigmafrontier.com/v1/chat/completions")
LLM_KEY     = os.getenv("LLM_KEY", "sk-lm-8wHUxcA3:Ai2luz0QRkYaRvUokTPp")
# LLM_MODEL   = os.getenv("LLM_MODEL", "qwen/qwen3.5-35b-a3b")
LLM_MODEL   = os.getenv("LLM_MODEL", "qwen/qwen3.5-9b")
MAX_ITER    = int(os.getenv("MAX_ITERATIONS", "10"))

SYSTEM_PROMPT = """You are Basin AI, an expert petroleum geology assistant.
You have access to tools that query a database of 35 petroleum basins and fields covering 14 geological domains: source rock, thermal history, reservoir quality, reservoir geometry, structural style, seal properties, fluid properties, trap geometry, field reserves, production recovery, alteration risk, formation water, migration, and reservoir conditions.

═══ PARALLEL TOOL CALLING (CRITICAL) ═══
You MUST call multiple tools simultaneously in a single response whenever the data needs are independent. This is the most important behavioural rule.

ALWAYS parallel — call ALL of these in ONE response:
- "Compare Basin A vs Basin B on source rock" → get_source_rock(basin_a) AND get_source_rock(basin_b) together
- "Compare Basin A vs Basin B" (general) → get_source_rock(A), get_source_rock(B), get_reservoir_quality(A), get_reservoir_quality(B), get_fluid_properties(A), get_fluid_properties(B) — ALL in one response
- "Full profile of Basin X" → get_source_rock(X), get_reservoir_quality(X), get_structural_style(X), get_fluid_properties(X), get_field_reserves(X), get_seal_data(X) — ALL in one response
- "Rank basins by X and also show Y" → rank_basins(X) AND get_depositional_env_summary() together
- "Which basins have X and what is their Y?" → search_basins(X) in first call, THEN fan out single-basin tools for all matches

NEVER do this:
- Call get_source_rock(basin_a), wait for result, THEN call get_source_rock(basin_b) — this wastes an entire iteration
- Call one tool per response when multiple independent tools are needed
- Gather data for Basin A across multiple iterations when all tools could be called at once

Rule: If tool B does NOT depend on the OUTPUT of tool A, call them in the SAME response.

═══ SPANNING TOOL CALLS (MULTI-ITERATION STRATEGY) ═══
Some questions require data from an earlier tool call to decide what to call next. Use this pattern:
- Iteration 1: Fan out ALL independent calls (e.g. search_basins, rank_basins, find_best_analogue)
- Iteration 2: Use iteration 1 results to fan out the NEXT batch of independent calls (e.g. get details for each basin returned)
- Iteration 3: Synthesise and answer, or call comparison tools if needed

Example — "Which basins have Type II kerogen and how do their reservoirs compare?"
  Iteration 1: search_basins(query="Type II kerogen") — must run first to get basin names
  Iteration 2: get_reservoir_quality(basin_1), get_reservoir_quality(basin_2), get_reservoir_quality(basin_3) — ALL parallel
  Iteration 3: Final answer synthesising the comparison

Aim for 2–3 iterations maximum. Every unnecessary iteration is wasted time.

═══ ENRICHMENT RULES (apply BEFORE calling tools) ═══
When the user's question is OPEN-ENDED or missing basin names, enrich it yourself:
1. COMPARISON with 0 basins specified:
   → Choose 2 geologically contrasting basins that make the comparison interesting.
   → State which basins you chose and WHY.
2. COMPARISON with 1 basin specified:
   → Call find_best_analogue for that basin, then use the top-scoring analogue as the second basin.
3. PROFILE with 0 basins specified:
   → Ask yourself: is the user asking about ALL basins (aggregation) or a specific one?
   → If aggregation: use get_all_basins, get_maturity_distribution, get_depositional_env_summary, or rank_basins.
   → If unclear: pick a representative basin and state your choice.
4. SCREENING ("which basins have X?"):
   → Use search_basins with the appropriate query and filters.
5. RANKING ("highest/best/most"):
   → Use rank_basins with the appropriate metric.

═══ TOOL SELECTION GUIDE ═══
SINGLE BASIN queries:
- Source rock quality/kerogen/TOC/maturity → get_source_rock
- Thermal history/heat flow/burial → get_thermal_history
- Reservoir porosity/perm/NTG/depth → get_reservoir_quality
- Sand body geometry/stacking/facies → get_reservoir_geometry
- Structural style/traps/faults/tectonics → get_structural_style
- Seal lithology/thickness/Pc/tested → get_seal_data
- API gravity/GOR/HC phase/viscosity → get_fluid_properties
- Trap closure/spill point/column height → get_trap_geometry
- Field reserves/discoveries/field count → get_field_reserves
- Recovery factor/OOIP/production/IOR → get_production_recovery
- Biodegradation/H2S/gas washing → get_alteration_risk
- Salinity/water type/Rw → get_formation_water
- Migration distance/carrier beds/kitchen → get_migration
- Pressure/temperature/contacts/aquifer → get_reservoir_conditions

TWO BASIN comparisons:
- Source rock → compare_source_rocks (includes overlap flags)
- Thermal → compare_thermal_history (includes deltas)
- Structure → compare_structural_styles (includes match flags)
- For other domains: call the single-basin tool for EACH basin IN THE SAME RESPONSE, then compare yourself.

CROSS-BASIN operations:
- HC consistency check → assess_hc_consistency (single) or loop for all
- Find analogues → find_best_analogue
- Rank by metric → rank_basins
- Maturity distribution → get_maturity_distribution
- Depositional environments → get_depositional_env_summary
- Find basins by criteria → search_basins

═══ OUTPUT RULES ═══
1. ALWAYS cite the data_source field from tool results verbatim.
2. Present numeric ranges as "X–Y (mid: Z)" format.
3. NEVER invent or fabricate data not returned by tools.
4. When data is at basin-level ranges (not per-well), state: "Note: This data represents basin-level screening ranges."
5. For multi-value arrays (kerogen types, depositional environments), list all values.
6. Structure long answers with clear headings.
7. If a tool returns an error, state it and try an alternative approach.
8. When you have gathered sufficient data, provide your final answer directly.

═══ STOP CONDITIONS (CRITICAL — FOLLOW STRICTLY) ═══
After your tool calls return data:
1. If a domain tool (get_source_rock, get_reservoir_quality, etc.) returned valid JSON with basin data, that basin WAS resolved. Do NOT call resolve_basin for it.
2. Do NOT call search_basins or get_all_basins to verify basins that already returned data.
3. Do NOT call resolve_basin at all — every domain tool resolves basin names internally via fuzzy matching.
4. Once you have data covering all domains the user asked about, STOP and write your final answer immediately.
Unnecessary verification calls waste time and provide no new information."""


# ── Async LLM Client ────────────────────────────────────────────────────
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAY = 10  # seconds to wait for model reload

# Module-level async HTTP client — reused across all requests
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Lazy-initialise a shared async HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=30.0),
        )
    return _http_client


async def call_llm(messages, tools=None, temperature=0.3, max_tokens=None):
    """Async LLM call with retry logic for model-unloaded scenarios.

    Args:
        messages:    Chat messages list.
        tools:       OpenAI-format tool definitions (None for final answer).
        temperature: Sampling temperature.
        max_tokens:  Token limit. Defaults to 1024 when tools are provided
                     (encourages tool calls over premature answers), 8192
                     when tools are None (final answer generation).
    """
    if max_tokens is None:
        max_tokens = 1024 if tools else 8192

    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
        body["parallel_tool_calls"] = True

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_KEY}",
    }

    client = _get_http_client()

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            r = await client.post(LLM_URL, json=body, headers=headers)

            if r.status_code == 200:
                return r.json()

            # ── Model unloaded: retry after non-blocking delay ──
            resp_text = r.text[:500]
            if r.status_code == 400 and "unload" in resp_text.lower():
                if attempt < LLM_MAX_RETRIES:
                    logger.warning(
                        f"Model unloaded (attempt {attempt}/{LLM_MAX_RETRIES}). "
                        f"Waiting {LLM_RETRY_DELAY}s for reload..."
                    )
                    await asyncio.sleep(LLM_RETRY_DELAY)
                    continue
                return {
                    "model_reloading": True,
                    "error": "Model is reloading. Please try again in a moment.",
                }

            return {"error": f"LLM HTTP {r.status_code}: {resp_text}"}

        except httpx.TimeoutException:
            return {
                "error": f"LLM timeout ({LLM_TIMEOUT}s). The model may be processing a large context.",
            }
        except httpx.RequestError as e:
            return {"error": str(e)}

    return {"error": "LLM request failed after retries."}


def strip_think_tags(text):
    """Remove <think>...</think> blocks emitted by reasoning models."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── MCP ↔ OpenAI Tool Format Conversion ─────────────────────────────────
def mcp_tools_to_openai(mcp_tools):
    """Convert MCP tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for tool in mcp_tools:
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
        if not schema:
            schema = {"type": "object", "properties": {}}
        clean_schema = _clean_schema(schema)
        desc = (tool.description or "")[:500]
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": desc,
                "parameters": clean_schema,
            },
        })
    return openai_tools


def _clean_schema(schema):
    """Strip non-essential JSON Schema keys that confuse smaller models."""
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    skip_keys = {"title", "additionalProperties", "$defs", "definitions", "examples"}
    for key, value in schema.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            cleaned[key] = value
    if "properties" in cleaned and "type" not in cleaned:
        cleaned["type"] = "object"
    return cleaned


def extract_tool_result(result):
    """Extract text content from an MCP tool result."""
    if not result or not result.content:
        return "{}"
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(str(block.data))
        else:
            parts.append(str(block))
    return "\n".join(parts)


# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(title="Basin AI Insights", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_http_client():
    """Close the shared httpx client on app shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


@app.get("/")
async def serve_index():
    return FileResponse("index.html", media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "mcp_url": MCP_SSE_URL}


@app.get("/api/chat")
async def chat_sse(request: Request, question: str):
    """
    SSE endpoint: streams agentic workflow events as the orchestrator runs.
    Connect via EventSource: /api/chat?question=...
    """

    async def event_generator():
        try:
            yield {"event": "status", "data": json.dumps({
                "message": "Connecting to MCP server...",
                "model": LLM_MODEL,
            })}

            async with sse_client(MCP_SSE_URL) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools_response = await session.list_tools()
                    mcp_tools = tools_response.tools
                    openai_tools = mcp_tools_to_openai(mcp_tools)

                    yield {"event": "status", "data": json.dumps({
                        "message": f"Connected. Loaded {len(openai_tools)} tools.",
                    })}

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ]

                    final_answer = None

                    # ── Execute a single MCP tool call ──
                    async def execute_one(tc_id, func_name, func_args):
                        try:
                            result = await session.call_tool(func_name, func_args)
                            result_text = extract_tool_result(result)
                        except Exception as e:
                            result_text = json.dumps({"error": str(e)})
                        if len(result_text) > 4000:
                            result_text = result_text[:4000] + "\n... [TRUNCATED]"
                        return tc_id, func_name, func_args, result_text

                    for iteration in range(1, MAX_ITER + 1):
                        # ── Client disconnect check ──
                        if await request.is_disconnected():
                            logger.info("Client disconnected, stopping.")
                            return

                        yield {"event": "iteration", "data": json.dumps({
                            "number": iteration,
                            "max": MAX_ITER,
                        })}

                        # ── Call LLM (async, non-blocking) ──
                        t0 = time.time()
                        response = await call_llm(messages, tools=openai_tools)
                        llm_time = round(time.time() - t0, 1)

                        if "error" in response:
                            # ── Model reloading: wait and retry once ──
                            if response.get("model_reloading"):
                                yield {"event": "status", "data": json.dumps({
                                    "message": "Model was unloaded. Reloading — this may take a moment...",
                                })}
                                await asyncio.sleep(15)
                                response = await call_llm(messages, tools=openai_tools)
                                llm_time = round(time.time() - t0, 1)
                                if "error" in response:
                                    yield {"event": "error", "data": json.dumps({
                                        "message": response["error"],
                                    })}
                                    return
                            else:
                                yield {"event": "error", "data": json.dumps({
                                    "message": response["error"],
                                })}
                                return

                        choice = response.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls", [])

                        # ── Stream LLM thinking if present alongside tool calls ──
                        thinking = strip_think_tags(content)
                        if thinking and tool_calls:
                            yield {"event": "thinking", "data": json.dumps({
                                "content": thinking,
                                "llm_time_s": llm_time,
                            })}

                        if tool_calls:
                            # ── Parse all tool calls from this turn ──
                            calls_summary = []
                            parsed = []
                            for tc in tool_calls:
                                func_name = tc["function"]["name"]
                                func_args_raw = tc["function"].get("arguments", "{}")
                                tc_id = tc.get("id", f"call_{iteration}_{func_name}")
                                try:
                                    func_args = (
                                        json.loads(func_args_raw)
                                        if isinstance(func_args_raw, str)
                                        else func_args_raw
                                    )
                                except json.JSONDecodeError:
                                    func_args = {}
                                parsed.append((tc_id, func_name, func_args))
                                calls_summary.append({
                                    "id": tc_id,
                                    "name": func_name,
                                    "args": func_args,
                                })

                            yield {"event": "tool_calls", "data": json.dumps({
                                "iteration": iteration,
                                "count": len(tool_calls),
                                "calls": calls_summary,
                                "llm_time_s": llm_time,
                            })}

                            # Append assistant message with tool_calls to history
                            messages.append(message)

                            # ── Execute ALL tool calls concurrently ──
                            results = await asyncio.gather(
                                *[execute_one(tc_id, fn, fa) for tc_id, fn, fa in parsed]
                            )

                            # ── Emit each result and append to message history ──
                            for tc_id, func_name, func_args, result_text in results:
                                try:
                                    result_parsed = json.loads(result_text)
                                except (json.JSONDecodeError, TypeError):
                                    result_parsed = result_text

                                yield {"event": "tool_result", "data": json.dumps({
                                    "id": tc_id,
                                    "name": func_name,
                                    "args": func_args,
                                    "result": result_parsed,
                                })}

                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": result_text,
                                })
                        else:
                            # ── No tool calls: this is the final answer turn ──
                            finish_reason = choice.get("finish_reason", "")
                            final_text = strip_think_tags(content)

                            # If truncated (hit 1024-token cap during tool iteration),
                            # re-call with full budget and no tools
                            if finish_reason == "length":
                                logger.info(
                                    "Answer truncated at 1024 tokens. "
                                    "Re-calling with max_tokens=8192, no tools..."
                                )
                                yield {"event": "status", "data": json.dumps({
                                    "message": "Generating full answer...",
                                })}
                                t0_final = time.time()
                                final_response = await call_llm(
                                    messages, tools=None, max_tokens=8192,
                                )
                                final_llm_time = round(time.time() - t0_final, 1)
                                if "error" not in final_response:
                                    final_choice = final_response.get("choices", [{}])[0]
                                    final_answer = strip_think_tags(
                                        final_choice.get("message", {}).get("content", "")
                                    )
                                    llm_time = final_llm_time
                                else:
                                    final_answer = final_text
                            else:
                                final_answer = final_text

                            yield {"event": "answer", "data": json.dumps({
                                "content": final_answer,
                                "iterations_used": iteration,
                                "llm_time_s": llm_time,
                            })}
                            break

                    # ── Force answer if agentic loop exhausted ──
                    if final_answer is None:
                        yield {"event": "status", "data": json.dumps({
                            "message": "Max iterations reached. Generating final answer...",
                        })}
                        messages.append({
                            "role": "user",
                            "content": (
                                "You have reached the maximum number of tool calls. "
                                "Based on ALL the data you have gathered so far, provide your "
                                "final comprehensive answer NOW. Do not call any more tools."
                            ),
                        })
                        response = await call_llm(messages, tools=None)
                        if "error" not in response:
                            choice = response.get("choices", [{}])[0]
                            final_answer = strip_think_tags(
                                choice.get("message", {}).get("content", "")
                            )
                            yield {"event": "answer", "data": json.dumps({
                                "content": final_answer,
                                "iterations_used": MAX_ITER,
                                "forced": True,
                            })}
                        else:
                            yield {"event": "error", "data": json.dumps({
                                "message": "Could not produce final answer after max iterations.",
                            })}

            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except Exception as e:
            logger.exception("SSE stream error")
            yield {"event": "error", "data": json.dumps({
                "message": str(e),
            })}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)