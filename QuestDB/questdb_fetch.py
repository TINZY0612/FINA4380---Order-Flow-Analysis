"""
Pull data from QuestDB into Python (pandas) for analysis.

Requirements: pip install requests pandas

Usage:
  from questdb_fetch import query_questdb, fetch_l2_orderbook
  df = query_questdb("SELECT * FROM l2_orderbook LIMIT 1000")
  df = fetch_l2_orderbook(symbol="BTCUSDT", limit=10_000)

Alternative for very large result sets: use Postgres wire (port 8812) with
  pip install psycopg2-binary
  conn = psycopg2.connect(host='127.0.0.1', port=8812, user='admin', password='quest', dbname='qdb')
  df = pd.read_sql("SELECT * FROM l2_orderbook", conn)
"""
import urllib.parse
from typing import Optional

import pandas as pd
import requests

# Default: same host as ccxt_l2_to_questdb.py
QUESTDB_HOST = "http://localhost:9000"


def query_questdb(
    sql: str,
    host: str = QUESTDB_HOST,
) -> pd.DataFrame:
    """
    Run a SQL query on QuestDB and return the result as a pandas DataFrame.

    Parameters
    ----------
    sql : str
        SQL query (e.g. SELECT ... FROM l2_orderbook WHERE ...).
    host : str
        Base URL of QuestDB (default http://localhost:9000).

    Returns
    -------
    pd.DataFrame
        Query result with columns and dtypes set. Timestamp columns are parsed to datetime64.
    """
    url = f"{host.rstrip('/')}/exec"
    params = {"query": sql}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"QuestDB error: {data.get('error', data)}")

    columns = data.get("columns", [])
    dataset = data.get("dataset", [])
    if not columns:
        return pd.DataFrame()

    col_names = [c["name"] for c in columns]
    col_types = {c["name"]: c.get("type", "STRING") for c in columns}
    df = pd.DataFrame(dataset, columns=col_names)

    # Parse timestamp column(s) to datetime
    for name in col_names:
        if col_types.get(name) == "TIMESTAMP" and name in df.columns:
            df[name] = pd.to_datetime(df[name], utc=True)

    return df


def fetch_l2_orderbook(
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
    host: str = QUESTDB_HOST,
) -> pd.DataFrame:
    """
    Load L2 orderbook table from QuestDB into a DataFrame.

    Parameters
    ----------
    symbol : str, optional
        Filter by symbol (e.g. "BTCUSDT"). If None, all symbols are returned.
    limit : int, optional
        Max rows (default 100_000). Use None for no limit (can be slow on huge tables).
    host : str
        QuestDB base URL.

    Returns
    -------
    pd.DataFrame
        Table with timestamp, symbol, bid_1_p, bid_1_s, ..., ask_N_p, ask_N_s.
    """
    conditions = []
    if symbol:
        # Escape single quotes in symbol
        safe = symbol.replace("'", "''")
        conditions.append(f"symbol = '{safe}'")
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
    sql = f"SELECT * FROM l2_orderbook{where} ORDER BY timestamp{limit_clause}"
    return query_questdb(sql, host=host)


if __name__ == "__main__":
    # Example: fetch last 1000 rows for BTC
    df = fetch_l2_orderbook(symbol="BTCUSDT", limit=1000)
    print(df.shape)
    print(df.head())
