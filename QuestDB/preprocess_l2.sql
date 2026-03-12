-- Preprocess L2 data in QuestDB to match pipeline_rank_ic.py steps:
-- 1) Align timestamps to 100ms grid (one row per symbol per bucket)
-- 2) Gap-fill so consecutive timestamps are 100ms (per symbol)
-- 3) Align all symbols to the same timestamps (master = symbol with smallest count)
--
-- Run in QuestDB web console (http://localhost:9000) or via /exec API.
-- Table: l2_orderbook (timestamp, symbol, bid_1_p, bid_1_s, ..., ask_20_p, ask_20_s)

-- =============================================================================
-- STEP 1: Grid alignment (100ms) + one row per (symbol, grid_ts)
-- =============================================================================
-- Floor timestamp to 100ms and pick one row per bucket (e.g. first by timestamp).
-- QuestDB: use timestamp_floor with 'T' = millisecond, 100 = stride.
-- If timestamp_floor('T', ts, 100) is not supported, try the SAMPLE BY variant below.

WITH grid_raw AS (
  SELECT
    symbol,
    timestamp_floor('T', timestamp, 100) AS ts,
    first(bid_1_p)  AS bid_1_p,  first(bid_1_s)  AS bid_1_s,
    first(bid_2_p)  AS bid_2_p,  first(bid_2_s)  AS bid_2_s,
    first(bid_3_p)  AS bid_3_p,  first(bid_3_s)  AS bid_3_s,
    first(bid_4_p)  AS bid_4_p,  first(bid_4_s)  AS bid_4_s,
    first(bid_5_p)  AS bid_5_p,  first(bid_5_s)  AS bid_5_s,
    first(bid_6_p)  AS bid_6_p,  first(bid_6_s)  AS bid_6_s,
    first(bid_7_p)  AS bid_7_p,  first(bid_7_s)  AS bid_7_s,
    first(bid_8_p)  AS bid_8_p,  first(bid_8_s)  AS bid_8_s,
    first(bid_9_p)  AS bid_9_p,  first(bid_9_s)  AS bid_9_s,
    first(bid_10_p) AS bid_10_p, first(bid_10_s) AS bid_10_s,
    first(bid_11_p) AS bid_11_p, first(bid_11_s) AS bid_11_s,
    first(bid_12_p) AS bid_12_p, first(bid_12_s) AS bid_12_s,
    first(bid_13_p) AS bid_13_p, first(bid_13_s) AS bid_13_s,
    first(bid_14_p) AS bid_14_p, first(bid_14_s) AS bid_14_s,
    first(bid_15_p) AS bid_15_p, first(bid_15_s) AS bid_15_s,
    first(bid_16_p) AS bid_16_p, first(bid_16_s) AS bid_16_s,
    first(bid_17_p) AS bid_17_p, first(bid_17_s) AS bid_17_s,
    first(bid_18_p) AS bid_18_p, first(bid_18_s) AS bid_18_s,
    first(bid_19_p) AS bid_19_p, first(bid_19_s) AS bid_19_s,
    first(bid_20_p) AS bid_20_p, first(bid_20_s) AS bid_20_s,
    first(ask_1_p)  AS ask_1_p,  first(ask_1_s)  AS ask_1_s,
    first(ask_2_p)  AS ask_2_p,  first(ask_2_s)  AS ask_2_s,
    first(ask_3_p)  AS ask_3_p,  first(ask_3_s)  AS ask_3_s,
    first(ask_4_p)  AS ask_4_p,  first(ask_4_s)  AS ask_4_s,
    first(ask_5_p)  AS ask_5_p,  first(ask_5_s)  AS ask_5_s,
    first(ask_6_p)  AS ask_6_p,  first(ask_6_s)  AS ask_6_s,
    first(ask_7_p)  AS ask_7_p,  first(ask_7_s)  AS ask_7_s,
    first(ask_8_p)  AS ask_8_p,  first(ask_8_s)  AS ask_8_s,
    first(ask_9_p)  AS ask_9_p,  first(ask_9_s)  AS ask_9_s,
    first(ask_10_p) AS ask_10_p, first(ask_10_s) AS ask_10_s,
    first(ask_11_p) AS ask_11_p, first(ask_11_s) AS ask_11_s,
    first(ask_12_p) AS ask_12_p, first(ask_12_s) AS ask_12_s,
    first(ask_13_p) AS ask_13_p, first(ask_13_s) AS ask_13_s,
    first(ask_14_p) AS ask_14_p, first(ask_14_s) AS ask_14_s,
    first(ask_15_p) AS ask_15_p, first(ask_15_s) AS ask_15_s,
    first(ask_16_p) AS ask_16_p, first(ask_16_s) AS ask_16_s,
    first(ask_17_p) AS ask_17_p, first(ask_17_s) AS ask_17_s,
    first(ask_18_p) AS ask_18_p, first(ask_18_s) AS ask_18_s,
    first(ask_19_p) AS ask_19_p, first(ask_19_s) AS ask_19_s,
    first(ask_20_p) AS ask_20_p, first(ask_20_s) AS ask_20_s
  FROM l2_orderbook
  GROUP BY symbol, timestamp_floor('T', timestamp, 100)
)
SELECT * FROM grid_raw ORDER BY symbol, ts;

-- =============================================================================
-- ALTERNATIVE STEP 1 (if timestamp_floor('T', ts, 100) is not available)
-- =============================================================================
-- QuestDB stores timestamp in microseconds by default. 100ms = 100_000 us.
-- Floor to 100ms: bucket_us = (timestamp / 100000L) * 100000L
-- Then use cast(bucket_us as timestamp) if needed, or keep as long for grouping.
--
-- Simplified grid (one row per symbol per 100ms) using long division:
--
-- WITH grid_raw AS (
--   SELECT symbol, (timestamp / 100000) * 100000 AS ts,
--          first(bid_1_p) AS bid_1_p, first(bid_1_s) AS bid_1_s, ...
--   FROM l2_orderbook
--   GROUP BY symbol, (timestamp / 100000) * 100000
-- )
-- SELECT * FROM grid_raw ORDER BY symbol, ts;

-- =============================================================================
-- STEP 2: Gap fill (100ms grid per symbol)
-- =============================================================================
-- QuestDB: use SAMPLE BY 100T FILL(PREV) when querying a *single* symbol.
-- Example for one symbol (no cross-symbol gap fill in one query):

-- SELECT timestamp, symbol, bid_1_p, bid_1_s, ..., ask_20_p, ask_20_s
-- FROM l2_orderbook
-- WHERE symbol = 'BTCUSDT'
-- SAMPLE BY 100T FILL(PREV);

-- For all symbols with gap fill, you either:
-- - Run the above per symbol and UNION ALL, or
-- - Use the Python pipeline (recommended for full gap fill).

-- =============================================================================
-- STEP 3: Align all symbols to master (same row count)
-- =============================================================================
-- Master = symbol with smallest number of rows. Then keep only those timestamps for every symbol.

-- 3a) Symbol with smallest count (use result as master_symbol):
-- SELECT symbol, count() AS n FROM l2_orderbook GROUP BY symbol ORDER BY n ASC LIMIT 1;

-- 3b) Timestamps of the master (replace 'AVAXUSDT' with result from 3a):
-- SELECT timestamp FROM l2_orderbook WHERE symbol = 'AVAXUSDT' ORDER BY timestamp;

-- 3c) Restrict all symbols to master timestamps (run after step 1 + 2 are materialized):
-- WITH master_ts AS (SELECT DISTINCT timestamp FROM l2_orderbook WHERE symbol = 'AVAXUSDT'),
--      grid AS ( ... your grid-aligned + gap-filled result ... )
-- SELECT g.* FROM grid g
-- WHERE g.timestamp IN (SELECT timestamp FROM master_ts)
-- ORDER BY g.symbol, g.timestamp;
