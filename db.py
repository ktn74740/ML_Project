import sqlite3
import pandas as pd
from typing import Tuple

def connect_db(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return pd.read_sql_query(q, conn, params=(table_name,)).shape[0] > 0

def write_replace(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str) -> None:
    """Replace table contents with df."""
    df.to_sql(table_name, conn, if_exists="replace", index=False)

def read_sql(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)