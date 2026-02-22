#!/usr/bin/env python3
"""Decision Arena: mission-driven strategic simulator powered by OpenGradient."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import opengradient as og


DEFAULT_MODEL_PRIORITY = [
    ("GPT_4O", og.TEE_LLM.GPT_4O),
    ("CLAUDE_3_5_HAIKU", og.TEE_LLM.CLAUDE_3_5_HAIKU),
    ("GEMINI_2_0_FLASH", og.TEE_LLM.GEMINI_2_0_FLASH),
]


@dataclass
class DecisionClient:
    client: og.Client
    model_name: str
    model: Any


@dataclass
class DecisionResult:
    model_name: str
    decision_frame: str
    challenge_brief: str
    mission_brief: str
    mission_board: List[Dict[str, Any]]
    scenario_map: Dict[str, Any]
    execution_plan: List[str]
    boss_fight: Dict[str, Any]
    verdict: Dict[str, Any]
    raw_output: str


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _required_approval_amount() -> float:
    try:
        return float(os.getenv("OPG_APPROVAL_AMOUNT", "1.0"))
    except Exception:
        return 1.0


def _is_payment_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "402" in msg or "payment required" in msg


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    hints = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "server disconnected",
        "connection reset",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(h in msg for h in hints)


def _configured_settlement_modes() -> List[Tuple[str, Any]]:
    raw = os.getenv("DA_SETTLEMENT_MODES", "SETTLE,SETTLE_BATCH")
    names = [piece.strip().upper() for piece in raw.split(",") if piece.strip()]
    if not names:
        names = ["SETTLE", "SETTLE_BATCH"]

    out: List[Tuple[str, Any]] = []
    for name in names:
        mode = getattr(og.x402SettlementMode, name, None)
        if mode is not None:
            out.append((name, mode))

    if not out and hasattr(og.x402SettlementMode, "SETTLE_BATCH"):
        out = [("SETTLE_BATCH", og.x402SettlementMode.SETTLE_BATCH)]
    return out


def _model_candidates(current_name: str, current_model: Any) -> List[Tuple[str, Any]]:
    ordered: List[Tuple[str, Any]] = [(current_name, current_model)]
    ordered.extend(DEFAULT_MODEL_PRIORITY)

    seen: set[str] = set()
    out: List[Tuple[str, Any]] = []
    for name, model in ordered:
        key = str(model)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, model))

    max_candidates = _env_int("DA_MODEL_FALLBACK_CANDIDATES", 4, 1, 8)
    return out[:max_candidates]


def load_private_key() -> Optional[str]:
    def _clean_key(raw: str | None) -> str:
        if not raw:
            return ""
        value = raw.strip().strip('"').strip("'")
        if re.fullmatch(r"[0-9a-fA-F]{64}", value):
            value = f"0x{value}"
        return value

    env_candidates = (
        "OG_PRIVATE_KEY",
        "OPENGRADIENT_PRIVATE_KEY",
        "OPEN_GRADIENT_PRIVATE_KEY",
        "OPG_PRIVATE_KEY",
        "WALLET_PRIVATE_KEY",
        "PRIVATE_KEY",
    )
    for env_name in env_candidates:
        key = _clean_key(os.getenv(env_name))
        if key:
            return key

    # Lightweight .env reader (no external dependency).
    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key_name = k.strip()
                if key_name not in {
                    "OG_PRIVATE_KEY",
                    "OPENGRADIENT_PRIVATE_KEY",
                    "OPEN_GRADIENT_PRIVATE_KEY",
                    "OPG_PRIVATE_KEY",
                    "WALLET_PRIVATE_KEY",
                    "PRIVATE_KEY",
                }:
                    continue
                value = _clean_key(v)
                if value:
                    return value
        except Exception:
            continue

    cfg_path = Path.home() / ".opengradient_config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            cfg_key = data.get("private_key")
            if isinstance(cfg_key, str):
                value = _clean_key(cfg_key)
                if value:
                    return value
        except Exception:
            return None
    return None


def _resolve_model(force_model: Optional[str]) -> Tuple[str, Any]:
    if force_model:
        for name, model in DEFAULT_MODEL_PRIORITY:
            if force_model == name or force_model == str(model):
                return name, model
        return force_model, force_model

    default_model = os.getenv("DA_DEFAULT_MODEL", "").strip()
    if default_model:
        for name, model in DEFAULT_MODEL_PRIORITY:
            if default_model == name or default_model == str(model):
                return name, model

    return DEFAULT_MODEL_PRIORITY[0]


def init_decision_client(force_model: Optional[str] = None) -> DecisionClient:
    private_key = load_private_key()
    if not private_key:
        raise RuntimeError(
            "Missing OG_PRIVATE_KEY (or OPENGRADIENT_PRIVATE_KEY) and "
            "~/.opengradient_config.json private_key"
        )

    client = og.Client(private_key=private_key)

    if hasattr(client.llm, "ensure_opg_approval") and _is_truthy_env(os.getenv("OPG_AUTO_APPROVE", "1")):
        try:
            client.llm.ensure_opg_approval(opg_amount=_required_approval_amount())
        except Exception as exc:
            eprint(f"[warn] ensure_opg_approval failed: {exc}")

    model_name, model = _resolve_model(force_model)
    return DecisionClient(client=client, model_name=model_name, model=model)


def _chat_once(
    dec: DecisionClient,
    *,
    model: Any,
    settlement_mode: Any,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    result = dec.client.llm.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        x402_settlement_mode=settlement_mode,
    )
    return ((result.chat_output or {}).get("content") or "").strip()


def chat(dec: DecisionClient, system: str, user: str, max_tokens: int = 900, temperature: float = 0.2) -> str:
    retries = _env_int("DA_CHAT_RETRIES", 2, 1, 8)
    base_backoff = _env_float("DA_CHAT_RETRY_BASE_SEC", 0.25, 0.05, 4.0)
    max_backoff = _env_float("DA_CHAT_RETRY_MAX_SEC", 1.2, 0.1, 10.0)

    settlement_modes = _configured_settlement_modes()
    candidates = _model_candidates(dec.model_name, dec.model)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        for candidate_name, candidate_model in candidates:
            for _, mode in settlement_modes:
                try:
                    text = _chat_once(
                        dec,
                        model=candidate_model,
                        settlement_mode=mode,
                        system=system,
                        user=user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    dec.model_name = candidate_name
                    dec.model = candidate_model
                    return text
                except Exception as exc:
                    last_exc = exc
                    if not (_is_payment_error(exc) or _is_retryable_error(exc)):
                        raise

        if attempt < retries:
            sleep_for = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
            time.sleep(sleep_for)

    if last_exc is not None and _is_payment_error(last_exc):
        raise RuntimeError(
            "OpenGradient returned HTTP 402 after trying model/settlement fallbacks. "
            "Fund backend wallet with more OPG or reduce output/token limits. "
            f"Original error: {last_exc}"
        ) from last_exc

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Decision chat failed with no response.")


def _extract_json(raw: str) -> Optional[dict]:
    text = raw.strip()
    if not text:
        return None

    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            piece = chunk.strip()
            if piece.startswith("json"):
                piece = piece[4:].strip()
            if piece.startswith("{") and piece.endswith("}"):
                try:
                    return json.loads(piece)
                except Exception:
                    pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _sanitize_mission_board(raw: object) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out

    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": str(item.get("id") or f"M{idx}"),
                "title": str(item.get("title") or "Untitled mission"),
                "goal": str(item.get("goal") or ""),
                "action": str(item.get("action") or ""),
                "difficulty": str(item.get("difficulty") or "medium"),
                "eta_days": int(item.get("eta_days") or 0),
                "success_metric": str(item.get("success_metric") or ""),
                "risk_guard": str(item.get("risk_guard") or ""),
            }
        )
    return out


def _sanitize_execution_plan(raw: object) -> List[str]:
    if not isinstance(raw, list):
        return []
    out = [str(item).strip() for item in raw if str(item).strip()]
    return out[:10]


def _sanitize_scenario_map(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "best_case": "",
            "base_case": "",
            "worst_case": "",
            "key_triggers": [],
            "early_warnings": [],
        }

    key_triggers_raw = raw.get("key_triggers")
    early_warnings_raw = raw.get("early_warnings")
    key_triggers = [str(x).strip() for x in key_triggers_raw] if isinstance(key_triggers_raw, list) else []
    early_warnings = [str(x).strip() for x in early_warnings_raw] if isinstance(early_warnings_raw, list) else []
    return {
        "best_case": str(raw.get("best_case") or ""),
        "base_case": str(raw.get("base_case") or ""),
        "worst_case": str(raw.get("worst_case") or ""),
        "key_triggers": key_triggers[:8],
        "early_warnings": early_warnings[:8],
    }


def _sanitize_verdict(raw: object, raw_output: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "verdict": "Unable to parse structured point of view.",
            "confidence_0_100": 0,
            "point_of_view": "",
            "why_this_view": "",
            "main_concerns": "",
            "opposite_view": "",
            "why_now": "",
            "why_not": "",
            "next_48h_actions": [],
            "next_30d_actions": [],
            "kpis": [],
            "red_lines": [],
            "contingency_plan": raw_output,
        }

    try:
        confidence = int(raw.get("confidence_0_100") or 0)
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))

    point_of_view = str(raw.get("point_of_view") or raw.get("verdict") or "").strip()
    why_this_view = str(raw.get("why_this_view") or raw.get("why_now") or "").strip()
    main_concerns = str(raw.get("main_concerns") or raw.get("why_not") or "").strip()
    opposite_view = str(raw.get("opposite_view") or raw.get("contingency_plan") or "").strip()

    return {
        "verdict": str(raw.get("verdict") or point_of_view or ""),
        "confidence_0_100": confidence,
        "point_of_view": point_of_view,
        "why_this_view": why_this_view,
        "main_concerns": main_concerns,
        "opposite_view": opposite_view,
        # Keep legacy keys for frontend/backward compatibility.
        "why_now": why_this_view,
        "why_not": main_concerns,
        "next_48h_actions": [],
        "next_30d_actions": [],
        "kpis": [],
        "red_lines": [],
        "contingency_plan": opposite_view,
    }


def run_decision_arena(
    dec: DecisionClient,
    *,
    decision: str,
    context: str,
    constraints: str,
    resources: str,
    risk_appetite: str,
    horizon_days: int,
    difficulty: str,
) -> DecisionResult:
    system = (
        "You are Decision Arena AI, an opinionated strategic advisor for real-world decisions. "
        "Return STRICT JSON only. Focus on a clear point of view with concise reasoning. "
        "Do not output step-by-step plans, 48h/30d actions, or time-based predictions."
    )

    user = (
        f"Decision:\n{decision}\n\n"
        f"Context:\n{context}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Resources:\n{resources}\n\n"
        f"Risk appetite: {risk_appetite}\n"
        f"Horizon days: {horizon_days}\n"
        f"Difficulty: {difficulty}\n\n"
        "Return JSON with keys:\n"
        "decision_frame (string, 4-6 lines),\n"
        "challenge_brief (string, main risks/failure modes),\n"
        "mission_brief (string, short summary),\n"
        "verdict (object with keys: verdict,confidence_0_100,point_of_view,why_this_view,main_concerns,opposite_view).\n"
        "Do NOT include next_48h_actions, next_30d_actions, execution plans, or predictions."
    )

    raw = chat(
        dec,
        system=system,
        user=user,
        max_tokens=_env_int("DA_MAX_TOKENS", 700, 128, 2600),
        temperature=0.15,
    )

    parsed = _extract_json(raw) or {}

    decision_frame = str(parsed.get("decision_frame") or "").strip()
    challenge_brief = str(parsed.get("challenge_brief") or "").strip()
    mission_brief = str(parsed.get("mission_brief") or "Point of view unavailable.").strip()
    mission_board = _sanitize_mission_board(parsed.get("mission_board"))
    scenario_map = _sanitize_scenario_map(parsed.get("scenario_map"))

    execution_plan = _sanitize_execution_plan(parsed.get("execution_plan"))

    boss_candidate = parsed.get("boss_fight")
    if isinstance(boss_candidate, dict):
        no_go_raw = boss_candidate.get("no_go_signals")
        no_go = [str(x).strip() for x in no_go_raw] if isinstance(no_go_raw, list) else []
        boss_fight = {
            "hardest_constraint": str(boss_candidate.get("hardest_constraint") or ""),
            "how_to_beat": str(boss_candidate.get("how_to_beat") or ""),
            "confidence_0_100": int(boss_candidate.get("confidence_0_100") or 0),
            "no_go_signals": no_go,
        }
    else:
        boss_fight = {
            "hardest_constraint": "",
            "how_to_beat": "",
            "confidence_0_100": 0,
            "no_go_signals": [],
        }
    verdict = _sanitize_verdict(parsed.get("verdict"), raw)

    if not mission_board:
        mission_board = []

    if not execution_plan:
        execution_plan = []
    if not decision_frame:
        decision_frame = "Decision frame unavailable. Add more context for a sharper point of view."
    if not challenge_brief:
        challenge_brief = "Challenge brief unavailable. Add clearer downside and risk details."

    return DecisionResult(
        model_name=dec.model_name,
        decision_frame=decision_frame,
        challenge_brief=challenge_brief,
        mission_brief=mission_brief,
        mission_board=mission_board,
        scenario_map=scenario_map,
        execution_plan=execution_plan,
        boss_fight=boss_fight,
        verdict=verdict,
        raw_output=raw,
    )


def available_models() -> Sequence[str]:
    return [name for name, _ in DEFAULT_MODEL_PRIORITY]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="decision_arena",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """\
            Decision Arena CLI

            Example:
              python decision_arena.py run "Should we launch in Q2?" --context "B2B SaaS" --constraints "small team"
            """
        ),
    )
    parser.add_argument("--model", default=None, help="Optional model name")

    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run mission-based decision simulation")
    run.add_argument("decision", help="Decision to evaluate")
    run.add_argument("--context", default="No extra context.")
    run.add_argument("--constraints", default="No explicit constraints.")
    run.add_argument("--resources", default="Unknown resources.")
    run.add_argument("--risk", default="balanced", choices=["low", "balanced", "high"])
    run.add_argument("--horizon", type=int, default=45)
    run.add_argument("--difficulty", default="hard", choices=["normal", "hard", "extreme"])
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        dec = init_decision_client(force_model=args.model)
        result = run_decision_arena(
            dec,
            decision=args.decision,
            context=args.context,
            constraints=args.constraints,
            resources=args.resources,
            risk_appetite=args.risk,
            horizon_days=max(7, min(args.horizon, 365)),
            difficulty=args.difficulty,
        )

        print(f"Model: {result.model_name}")
        print("=" * 72)
        print(result.decision_frame)
        print("\nChallenge Brief:")
        print(result.challenge_brief)
        print("\nMission Brief:")
        print(result.mission_brief)
        print("\nMission Board:")
        for mission in result.mission_board:
            print(f"- {mission['id']} | {mission['title']} | ETA {mission['eta_days']}d")
            print(f"  Goal: {mission['goal']}")
            print(f"  Action: {mission['action']}")
            print(f"  Metric: {mission['success_metric']}")
            print(f"  Guard: {mission['risk_guard']}")

        print("\nScenario Map:")
        print(f"Best: {result.scenario_map.get('best_case', '')}")
        print(f"Base: {result.scenario_map.get('base_case', '')}")
        print(f"Worst: {result.scenario_map.get('worst_case', '')}")
        print(f"Triggers: {result.scenario_map.get('key_triggers', [])}")
        print(f"Early warnings: {result.scenario_map.get('early_warnings', [])}")

        print("\nExecution Plan:")
        for idx, step in enumerate(result.execution_plan, start=1):
            print(f"{idx}. {step}")

        print("\nBoss Fight:")
        print(json.dumps(result.boss_fight, indent=2))
        print("\nVerdict:")
        print(json.dumps(result.verdict, indent=2))
        return 0
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
