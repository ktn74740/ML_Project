from db import connect_db, read_sql

DB_PATH = "covid_counties.db"


def run_query(query: str, params: tuple = ()):
    conn = connect_db(DB_PATH)
    try:
        result = read_sql(conn, query, params)
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    q = input("Enter SQL query: ").strip()

    if not q:
        print("No query provided.")
    else:
        try:
            df = run_query(q)
            print(df)
            print(f"\nRows returned: {len(df)}")
        except Exception as e:
            print(f"Error: {e}")




