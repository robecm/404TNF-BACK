import snowflake.connector
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )
    return conn


def get_snowflake_data():
    conn = get_snowflake_connection()
    cursor = conn.cursor()

    query = "SELECT TOP 10 * FROM HACKTEAM.RAW.TESS_LAND"
    cursor.execute(query)

    # Fetch all rows and return as a list of dicts
    result = cursor.fetchall()
    
    print(result)

get_snowflake_data()