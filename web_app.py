#!/usr/bin/env python3
"""Decision Arena Studio web app.

Tool 3 uses the same proven wallet-fee flow as tool 1, but the AI task is
mission-driven decision simulation.
"""

from __future__ import annotations

import os
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
from web3 import Web3

from decision_arena import (
    DecisionClient,
    available_models,
    init_decision_client,
    load_private_key,
    run_decision_arena,
)


APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "web_static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)

_CLIENT_CACHE: Dict[str, DecisionClient] = {}
_CLIENT_LOCK = threading.Lock()

BASE_SEPOLIA_CHAIN_ID_HEX = "0x14a34"
BASE_SEPOLIA_CHAIN_ID_INT = int(BASE_SEPOLIA_CHAIN_ID_HEX, 16)
BASE_SEPOLIA_RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")
OPG_TOKEN_ADDRESS = Web3.to_checksum_address(
    os.getenv("OPG_TOKEN_ADDRESS", "0x240b09731D96979f50B2C649C9CE10FcF9C7987F")
)
OPG_FEE_AMOUNT = Decimal(os.getenv("OPG_FEE_AMOUNT", "0.0001"))
OPG_FEE_WEI = int(OPG_FEE_AMOUNT * Decimal(10**18))
RUNS_PER_FEE_TX = max(1, int(os.getenv("RUNS_PER_FEE_TX", "10")))
FEE_TX_LOOKUP_TIMEOUT_SEC = float(os.getenv("FEE_TX_LOOKUP_TIMEOUT_SEC", "45"))
FEE_TX_LOOKUP_POLL_SEC = float(os.getenv("FEE_TX_LOOKUP_POLL_SEC", "1.5"))
USED_TX_TTL_SEC = max(60, int(os.getenv("USED_TX_TTL_SEC", "2592000")))
CREDITS_TTL_SEC = max(60, int(os.getenv("CREDITS_TTL_SEC", "2592000")))
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "decision_arena").strip() or "decision_arena"


def _clamp_text(value: str, max_len: int) -> str:
    return value[:max_len]


def _clean_env_value(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip().strip('"').strip("'")


def _first_env_value(*names: str) -> str:
    for name in names:
        value = _clean_env_value(os.getenv(name))
        if value:
            return value
    return ""


FEE_RECEIVER_ENV_NAMES = (
    "OPG_FEE_RECEIVER",
    "FEE_RECEIVER",
    "TREASURY_WALLET",
    "FEE_WALLET",
)
PRIVATE_KEY_ENV_NAMES = (
    "OG_PRIVATE_KEY",
    "OPENGRADIENT_PRIVATE_KEY",
    "OPEN_GRADIENT_PRIVATE_KEY",
    "OPG_PRIVATE_KEY",
    "WALLET_PRIVATE_KEY",
    "PRIVATE_KEY",
)


def _fee_config_flags() -> str:
    parts = []
    for name in FEE_RECEIVER_ENV_NAMES:
        parts.append(f"{name}={'1' if _clean_env_value(os.getenv(name)) else '0'}")
    for name in PRIVATE_KEY_ENV_NAMES:
        parts.append(f"{name}={'1' if _clean_env_value(os.getenv(name)) else '0'}")
    return ", ".join(parts)


def resolve_fee_receiver() -> str:
    raw = _first_env_value(*FEE_RECEIVER_ENV_NAMES)
    if raw:
        try:
            return Web3.to_checksum_address(raw)
        except Exception:
            # Invalid address text; continue and try deriving from private key.
            pass

    key = load_private_key()
    if key:
        try:
            acct = Web3().eth.account.from_key(key)
            return Web3.to_checksum_address(acct.address)
        except Exception:
            return ""
    return ""


W3_BASE = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex().lower()


class FeeStateStore:
    def __init__(self) -> None:
        self.redis_url = UPSTASH_REDIS_REST_URL
        self.redis_token = UPSTASH_REDIS_REST_TOKEN
        self._used_fee_tx: set[str] = set()
        self._wallet_run_credits: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        if self.redis_url and self.redis_token:
            return "redis"
        return "memory"

    def _wallet_key(self, wallet: str) -> str:
        return f"{REDIS_KEY_PREFIX}:credits:{wallet.lower()}"

    def _tx_key(self, tx_hash: str) -> str:
        return f"{REDIS_KEY_PREFIX}:tx:{tx_hash.lower()}"

    def _redis_call(self, *parts: str):
        resp = requests.post(
            self.redis_url,
            headers={
                "Authorization": f"Bearer {self.redis_token}",
                "Content-Type": "application/json",
            },
            json=list(parts),
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data.get("result") if isinstance(data, dict) else None

    def get_credits(self, wallet: str) -> int:
        if self.mode == "redis":
            raw = self._redis_call("GET", self._wallet_key(wallet))
            if raw is None:
                return 0
            try:
                return max(0, int(raw))
            except Exception:
                return 0

        with self._lock:
            return max(0, self._wallet_run_credits.get(wallet, 0))

    def consume_credit_if_any(self, wallet: str) -> tuple[bool, int]:
        if self.mode == "redis":
            key = self._wallet_key(wallet)
            remaining = int(self._redis_call("DECRBY", key, "1") or 0)
            if remaining >= 0:
                self._redis_call("EXPIRE", key, str(CREDITS_TTL_SEC))
                return True, remaining

            self._redis_call("INCRBY", key, "1")
            current = self.get_credits(wallet)
            if current < 0:
                self._redis_call("SET", key, "0", "EX", str(CREDITS_TTL_SEC))
                current = 0
            return False, current

        with self._lock:
            existing = self._wallet_run_credits.get(wallet, 0)
            if existing > 0:
                remaining = existing - 1
                self._wallet_run_credits[wallet] = remaining
                return True, remaining
            return False, 0

    def add_credits(self, wallet: str, amount: int) -> int:
        amount = max(0, int(amount))
        if self.mode == "redis":
            key = self._wallet_key(wallet)
            if amount > 0:
                remaining = int(self._redis_call("INCRBY", key, str(amount)) or 0)
            else:
                remaining = self.get_credits(wallet)
            self._redis_call("EXPIRE", key, str(CREDITS_TTL_SEC))
            return max(0, remaining)

        with self._lock:
            self._wallet_run_credits[wallet] = self._wallet_run_credits.get(wallet, 0) + amount
            return max(0, self._wallet_run_credits[wallet])

    def mark_tx_if_new(self, tx_hash: str) -> bool:
        if self.mode == "redis":
            result = self._redis_call("SET", self._tx_key(tx_hash), "1", "NX", "EX", str(USED_TX_TTL_SEC))
            return result == "OK"

        with self._lock:
            if tx_hash in self._used_fee_tx:
                return False
            self._used_fee_tx.add(tx_hash)
            return True

    def release_tx_mark(self, tx_hash: str) -> None:
        if self.mode == "redis":
            try:
                self._redis_call("DEL", self._tx_key(tx_hash))
            except Exception:
                pass
            return

        with self._lock:
            self._used_fee_tx.discard(tx_hash)


FEE_STATE = FeeStateStore()


def get_client(model: str | None = None) -> DecisionClient:
    key = (model or "").strip() or "__default__"
    with _CLIENT_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            return cached
        client = init_decision_client(force_model=model)
        _CLIENT_CACHE[key] = client
        return client


def _topic_addr(topic_hex: str) -> str:
    return Web3.to_checksum_address("0x" + topic_hex[-40:])


def _hex_to_int(raw: object) -> int:
    if isinstance(raw, bytes):
        return int.from_bytes(raw, byteorder="big")
    if hasattr(raw, "hex"):
        return int(raw.hex(), 16)
    if isinstance(raw, str):
        return int(raw, 16)
    return int(raw or 0)


def _wait_for_transaction(tx_hash: str):
    deadline = time.time() + FEE_TX_LOOKUP_TIMEOUT_SEC
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            tx = W3_BASE.eth.get_transaction(tx_hash)
            if tx:
                return tx
        except Exception as exc:
            last_exc = exc
        time.sleep(FEE_TX_LOOKUP_POLL_SEC)

    detail = f"last error: {last_exc}" if last_exc else "not propagated to RPC yet"
    raise ValueError(f"fee transaction not found after waiting {FEE_TX_LOOKUP_TIMEOUT_SEC:.0f}s ({detail})")


def _wait_for_receipt(tx_hash: str):
    deadline = time.time() + FEE_TX_LOOKUP_TIMEOUT_SEC
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            receipt = W3_BASE.eth.get_transaction_receipt(tx_hash)
            if receipt:
                return receipt
        except Exception as exc:
            last_exc = exc
        time.sleep(FEE_TX_LOOKUP_POLL_SEC)

    detail = f"last error: {last_exc}" if last_exc else "receipt unavailable"
    raise ValueError(f"fee receipt not found after waiting {FEE_TX_LOOKUP_TIMEOUT_SEC:.0f}s ({detail})")


def _require_fee_payment(payload: dict) -> tuple[str, str, int, bool]:
    wallet_address_raw = str(payload.get("wallet_address") or "").strip()
    fee_tx_hash_raw = str(payload.get("fee_tx_hash") or "").strip()
    if not wallet_address_raw:
        raise ValueError("wallet_address is required")
    fee_receiver = resolve_fee_receiver()
    if not fee_receiver:
        raise RuntimeError(
            "Fee configuration missing on server. "
            "Set OPG_FEE_RECEIVER/FEE_RECEIVER or OG_PRIVATE_KEY/OPENGRADIENT_PRIVATE_KEY. "
            f"Detected env flags: {_fee_config_flags()}"
        )

    wallet_address = Web3.to_checksum_address(wallet_address_raw)

    consumed, remaining = FEE_STATE.consume_credit_if_any(wallet_address)
    if consumed:
        return wallet_address, "", remaining, False

    if not fee_tx_hash_raw:
        raise ValueError("No remaining runs. fee_tx_hash is required")
    if not fee_tx_hash_raw.startswith("0x") or len(fee_tx_hash_raw) != 66:
        raise ValueError("fee_tx_hash must be a valid transaction hash")
    fee_tx_hash = fee_tx_hash_raw.lower()

    if not FEE_STATE.mark_tx_if_new(fee_tx_hash):
        raise ValueError("fee_tx_hash was already used")

    try:
        tx = _wait_for_transaction(fee_tx_hash)

        tx_from = Web3.to_checksum_address(tx["from"])
        if tx_from != wallet_address:
            raise ValueError("fee tx sender does not match wallet_address")
        if tx.get("chainId") and int(tx["chainId"]) != BASE_SEPOLIA_CHAIN_ID_INT:
            raise ValueError("fee tx is not on Base Sepolia")

        receipt = _wait_for_receipt(fee_tx_hash)
        if int(receipt.get("status", 0)) != 1:
            raise ValueError("fee transaction failed")

        paid_ok = False
        for log in receipt.get("logs", []):
            if Web3.to_checksum_address(log["address"]) != OPG_TOKEN_ADDRESS:
                continue
            topics = log.get("topics", [])
            if len(topics) < 3:
                continue
            t0 = topics[0].hex().lower()
            if t0 != TRANSFER_TOPIC:
                continue
            from_addr = _topic_addr(topics[1].hex())
            to_addr = _topic_addr(topics[2].hex())
            value_wei = _hex_to_int(log.get("data", "0x0"))
            if from_addr == wallet_address and to_addr == fee_receiver and value_wei >= OPG_FEE_WEI:
                paid_ok = True
                break

        if not paid_ok:
            raise ValueError("fee payment not found in transaction logs")
    except Exception:
        FEE_STATE.release_tx_mark(fee_tx_hash)
        raise

    remaining = FEE_STATE.add_credits(wallet_address, RUNS_PER_FEE_TX - 1)
    return wallet_address, fee_tx_hash, remaining, True


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/ping")
def api_ping():
    fee_receiver = resolve_fee_receiver()
    return jsonify(
        {
            "ok": True,
            "status": "ready",
            "tool": "Decision Arena Studio",
            "models": list(available_models()),
            "fee_required": True,
            "fee_token": OPG_TOKEN_ADDRESS,
            "fee_amount_opg": str(OPG_FEE_AMOUNT),
            "fee_receiver": fee_receiver,
            "fee_configured": bool(fee_receiver),
            "fee_chain_id": BASE_SEPOLIA_CHAIN_ID_HEX,
            "runs_per_fee_tx": RUNS_PER_FEE_TX,
            "credits_store": FEE_STATE.mode,
        }
    )


@app.route("/api/credits")
def api_credits():
    wallet_raw = str(request.args.get("wallet") or "").strip()
    if not wallet_raw:
        return jsonify({"ok": True, "remaining_runs": 0})
    try:
        wallet = Web3.to_checksum_address(wallet_raw)
        remaining = FEE_STATE.get_credits(wallet)
        return jsonify({"ok": True, "wallet_address": wallet, "remaining_runs": remaining})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    decision = _clamp_text(str(data.get("decision") or "").strip(), 900)
    context = _clamp_text(str(data.get("context") or "").strip(), 1800)
    constraints = _clamp_text(str(data.get("constraints") or "").strip(), 1400)
    resources = _clamp_text(str(data.get("resources") or "").strip(), 1400)
    risk_appetite = str(data.get("risk_appetite") or "balanced").strip().lower()
    horizon_days = int(data.get("horizon_days") or 45)
    horizon_days = max(7, min(horizon_days, 365))
    difficulty = str(data.get("difficulty") or "hard").strip().lower()
    model = str(data.get("model") or "").strip() or None

    if not decision:
        return jsonify({"ok": False, "error": "decision is required"}), 400

    try:
        wallet_address, fee_tx_hash, remaining_runs, paid_now = _require_fee_payment(data)
        dec = get_client(model)
        result = run_decision_arena(
            dec,
            decision=decision,
            context=context or "No additional context provided.",
            constraints=constraints or "No explicit constraints provided.",
            resources=resources or "Resources not specified.",
            risk_appetite=risk_appetite or "balanced",
            horizon_days=horizon_days,
            difficulty=difficulty or "hard",
        )

        print(
            f"[ANALYZE] model={result.model_name} wallet={wallet_address} "
            f"risk={risk_appetite} horizon={horizon_days} difficulty={difficulty}"
        )

        return jsonify(
            {
                "ok": True,
                "tool": "Decision Arena Studio",
                "model": result.model_name,
                "decision_frame": result.decision_frame,
                "challenge_brief": result.challenge_brief,
                "mission_brief": result.mission_brief,
                "mission_board": result.mission_board,
                "scenario_map": result.scenario_map,
                "execution_plan": result.execution_plan,
                "boss_fight": result.boss_fight,
                "verdict": result.verdict,
                "raw_output": result.raw_output,
                "wallet_address": wallet_address,
                "fee_tx_hash": fee_tx_hash,
                "remaining_runs": remaining_runs,
                "paid_now": paid_now,
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8099"))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 70)
    print("Decision Arena Studio")
    print(f"Open: http://{host}:{port}")
    print("=" * 70)
    app.run(host=host, port=port, debug=False)
