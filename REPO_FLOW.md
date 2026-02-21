# Repository flow: L2 data → factors → rank IC

## Current flow (as-is)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES (not unified)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Binance WebSocket (l1_ofi, l2_ofi, l2_obi)  → live stream, one factor each │
│ • CoinDesk REST (preliminiary)                 → historical minute snapshots  │
│ • CCXT → QuestDB (ccxt_l2_to_questdb)          → store only; no factor/IC     │
└─────────────────────────────────────────────────────────────────────────────┘
         │                      │                            │
         ▼                      ▼                            ▼
┌──────────────┐    ┌──────────────┐              ┌──────────────────────┐
│ OFI inline   │    │ OBI inline   │              │ questdb_fetch        │
│ (l2_ofi)     │    │ (l2_obi)     │              │ analyze_l2_example   │
└──────┬───────┘    └──────┬───────┘              │ (no factors / no IC) │
       │                    │                      └──────────────────────┘
       ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ Rank IC      │    │ Rank IC      │   (duplicated in each script)
│ (spearmanr)  │    │ (spearmanr)  │
└──────────────┘    └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STANDALONE FACTORS (not wired into any pipeline)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ factor2_microprice_deviation  │  factor3_depth_slope  │  factorl3_shannon_entropy │
│ Factor6_LiquidityDepletion   │  (OFI/OBI only inside l1_ofi, l2_ofi, l2_obi)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Issues:**
- Data format and source differ per script (WS vs REST vs QuestDB).
- Each “factor pipeline” (OFI, OBI, TFI) has its own data fetch + one factor + rank IC copy-paste.
- LOB factors (microprice, depth slope, entropy, depletion) are not used in any rank-IC pipeline.
- No single path: “get L2 from CCXT → compute all factors → rank IC”.

---

## Target flow (with factor library)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                                   │
│ • CCXT (poll or one-off)  →  • QuestDB (query)  →  • Binance WS (existing)   │
│   Same output: list of { timestamp, symbol, bids, asks }  (bids/asks = [[p,q]])│
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FACTOR LIBRARY (lob_factors)                                                 │
│ compute_all(bids, asks, n_levels) → { microprice_dev, depth_slope_imb,       │
│   entropy_imb, obi, bid_liq_5, ask_liq_5, ... }                            │
│ (Uses factor2, factor3, factorl3, plus OBI/liquidity in one place.)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE                                                                     │
│ 1. Load L2 snapshots (from CCXT loop, QuestDB, or WS buffer)                  │
│ 2. For each snapshot: mid_price + compute_all(...) → one row                │
│ 3. Build DataFrame: [timestamp, symbol, mid, f1, f2, ...]                    │
│ 4. Add forward return: ret_future = shift(mid, -k) / mid - 1                │
│ 5. Rank IC: spearmanr(factor, ret_future) per factor (and per symbol)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Improvements (summary)

| Area | Change |
|------|--------|
| **Factors** | Single `lob_factors` package: import factor2, factor3, factorl3, add OBI + liquidity; expose `compute_all(bids, asks, n_levels)` so one call gives every factor. |
| **Data** | Shared “L2 snapshot” format: `bids` / `asks` = list of `[price, size]`, best first. Adapters: CCXT fetch, QuestDB query, or Binance WS callback all produce this. |
| **Rank IC** | One function `rank_ic_report(df, factor_columns, ret_column)` that loops over factor columns and symbols and returns a small table (IC, p-value, count). |
| **Pipeline** | One script (e.g. `pipeline_rank_ic.py`) that: gets L2 from CCXT (or QuestDB), runs factors via library, builds DataFrame, adds forward return, runs rank IC report. |

---

## File layout (current)

```
FINA4380---Order-Flow-Analysis/
├── REPO_FLOW.md              # this file
├── lob_factors/               # factor library
│   __init__.py                # compute_all(), microprice_deviation, obi_weighted, etc.
├── pipeline_rank_ic.py        # L2 (CCXT or QuestDB) → factors → rank IC report
├── data_sources.py            # get_l2_snapshot_ccxt, collect_l2_ccxt, get_l2_from_questdb
├── factor2_microprice_deviation.py
├── factor3_depth_slope.py
├── factorl3_shannon_entropy.py
├── Factor6_LiquidityDepletion.py
├── l1_ofi.py, l2_ofi.py, l2_obi.py, tfi.py   # keep for live WS use
└── QuestDB/
    ccxt_l2_to_questdb.py, questdb_fetch.py, ...
```

**Run from repo root:**
- `python pipeline_rank_ic.py` — CCXT 60s, 200ms tick, then rank IC.
- `python pipeline_rank_ic.py --source questdb --limit 3000` — load from QuestDB, then rank IC.
- `python pipeline_rank_ic.py --source ccxt --duration 120 --tick 100` — longer live run.
