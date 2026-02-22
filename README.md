# Decision Arena Studio (Tool 3)

Fresh rebuild inspired by Tool 1 architecture:
- Same proven wallet fee + credits flow (Base Sepolia)
- New AI idea from Tool 2: mission-based strategic decision simulation

## What AI returns

For each run, the model generates:
- `decision_frame` (board-level framing)
- `challenge_brief` (red-team risks/failure modes)
- `mission_brief`
- `mission_board` (hard tasks with metrics/guards)
- `scenario_map` (best/base/worst + triggers + early warnings)
- `execution_plan` (concrete actions)
- `boss_fight` (hardest constraint + confidence + no-go signals)
- `verdict` (why now/why not, 48h + 30d actions, red lines, contingency)

## Requirements

- Python 3.10+
- `OG_PRIVATE_KEY` (or `OPENGRADIENT_PRIVATE_KEY`)
- Base Sepolia wallet in frontend (MetaMask)

## Install and run (local)

```bash
cd "/Users/rivale/Documents/New project/tool_workspace_3"
pip install -r requirements.txt
HOST=127.0.0.1 PORT=8099 python3 web_app.py
```

Open: `http://127.0.0.1:8099`

## Wallet/Fee flow (same style as Tool 1)

- Wallet connect is required before run.
- One payment of `0.0001 OPG` unlocks `10` runs.
- Default network: Base Sepolia.

## Important env vars

- `OG_PRIVATE_KEY` or `OPENGRADIENT_PRIVATE_KEY`
- `OPG_FEE_AMOUNT=0.0001`
- `RUNS_PER_FEE_TX=10`
- `OPG_FEE_RECEIVER=0x...` (optional; defaults to backend wallet)
- `OPG_AUTO_APPROVE=1`
- `OPG_APPROVAL_AMOUNT=1.0`
- `DA_DEFAULT_MODEL=GPT_4O`
- `DA_MAX_TOKENS=1200`
- `DA_CHAT_RETRIES=2`
- `DA_SETTLEMENT_MODES=SETTLE,SETTLE_BATCH`

## Deploy notes

- Vercel entrypoint: `api/index.py`
- `vercel.json` rewrites all routes to `/api/index`
- Railway start command is in `Procfile` and `railway.toml`
