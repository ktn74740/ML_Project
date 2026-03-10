from db import connect_db, write_replace
from data_loading import load_csv, clean_county_data

CSV_PATH = "us_counties_covid19_daily.csv"
DB_PATH = "covid_counties.db"
TABLE_NAME = "counties_daily"


def main():

    print("Starting ETL process...")

    # Load CSV
    raw_df = load_csv(CSV_PATH)
    print(f"CSV loaded: {len(raw_df):,} rows")

    # Clean data
    clean_df = clean_county_data(raw_df)
    print(f"Cleaned data: {len(clean_df):,} rows")

    # Connect to database
    conn = connect_db(DB_PATH)

    # Write data to database
    write_replace(conn, clean_df, TABLE_NAME)

    conn.close()

    print("Database created/updated successfully.")
    print(f"Database file: {DB_PATH}")
    print(f"Table name: {TABLE_NAME}")


if __name__ == "__main__":
    main()