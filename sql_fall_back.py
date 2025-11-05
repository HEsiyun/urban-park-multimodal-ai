from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
import os, pandas as pd
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any

def sql_fall_back(query: str) -> Dict[str, Any]:
    """Function to execute SQL query with fall back to a different LLM if needed."""
    # File paths
    DATA_DIR = os.path.abspath("data")
    PERMIT_XLSX = os.path.join(DATA_DIR, "4 Permits_2024.xlsx")
    PARK_XLSX = os.path.join(DATA_DIR, "parks.xlsx")
    FIELD_SIZE_XLSX = os.path.join(DATA_DIR, "3 vsfs_master_inventory_fieldsizes.xlsx")
    MAINTENANCE_TYPES_XLSX = os.path.join(DATA_DIR, "6 Maint Activity Types- Mar 2025.xlsx")
    MAINTENANCE_XLSX = os.path.join(DATA_DIR, "6 Stanley order list.xlsx")
    # switch to a file-backed SQLite DB
    DATABASE_PATH = os.path.join(DATA_DIR, "permits_data.sqlite3")
    PERMIT_SHEET = 0
    PARK_SHEET = 0
    DIAMOND_FIELD_SIZE_SHEET = 1
    RECTANGULAR_FIELD_SIZE_SHEET = 2
    MAINTENANCE_TYPES_SHEET = 0
    MAINTENANCE_SHEET = 0

    # Read Excel and normalize column names
    df = pd.read_excel(PERMIT_XLSX, sheet_name=PERMIT_SHEET)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')
        print(df['date'].head())
    
    df_park = pd.read_excel(PARK_XLSX, sheet_name=PARK_SHEET)
    df_park.columns = [col.lower().replace(" ", "_") for col in df_park.columns]

    df_field_size_diamond = pd.read_excel(FIELD_SIZE_XLSX, sheet_name=DIAMOND_FIELD_SIZE_SHEET)
    # df_field_size_diamond.fillna("None", inplace=True)
    df_field_size_diamond.columns = [col.lower().replace(" ", "_") for col in df_field_size_diamond.columns]
    df_field_size_rectangular = pd.read_excel(FIELD_SIZE_XLSX, sheet_name=RECTANGULAR_FIELD_SIZE_SHEET)
    # df_field_size_rectangular.fillna("None", inplace=True)
    df_field_size_rectangular.columns = [col.lower().replace(" ", "_") for col in df_field_size_rectangular.columns]
    
    df_maintenance_types = pd.read_excel(MAINTENANCE_TYPES_XLSX, sheet_name=MAINTENANCE_TYPES_SHEET)
    # df_maintenance_types.fillna("None", inplace=True)
    df_maintenance_types.columns = [col.lower().replace(" ", "_") for col in df_maintenance_types.columns]
    df_maintenance = pd.read_excel(MAINTENANCE_XLSX, sheet_name=MAINTENANCE_SHEET)
    # df_maintenance.fillna("None", inplace=True)
    df_maintenance.columns = [col.lower().replace(" ", "_") for col in df_maintenance.columns]


    # Use SQLAlchemy + SQLite: create/replace table on disk
    engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
    df.to_sql("event_data", con=engine, if_exists="replace", index=False)
    df_park.to_sql("park_basic_data", con=engine, if_exists="replace", index=False)
    df_field_size_diamond.to_sql("diamond_field_size_data", con=engine, if_exists="replace", index=False)
    df_field_size_rectangular.to_sql("rectangular_field_size_data", con=engine, if_exists="replace", index=False)
    df_maintenance_types.to_sql("maintenance_types_code_mapping", con=engine, if_exists="replace", index=False)
    df_maintenance.to_sql("maintenance_activity_data", con=engine, if_exists="replace", index=False)
    llm = ChatOllama(base_url="http://127.0.0.1:11434", model="llama3:8b", temperature=0, max_tokens=1000)

    # Connect LangChain to the file-backed SQLite via SQLAlchemy URI
    db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}")
    # table_schema = db.get_table_info(table_names=["event_data"])  # or build a string manually
    # result = db.run("SELECT SUM(total_costs) AS total_spent FROM (maintenance_activity_data JOIN maintenance_types_code_mapping ON activity_type = code) WHERE code = 100 GROUP BY code LIMIT 10;")
    # print("result:", result)
    # print("Events in June:", result)
    # prompt = f"Schema: {table_schema}\nQuestion: How many events occurred in June? SQL queries return results as lists of tuples. For example, [(5860,)] means the result is 5860."

    # Create an agent that can generate SQL and run it against the database
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description",
        return_intermediate_steps=True
    )


    # Use the agent
    resp = agent_executor.invoke(query)
    #response = agent_executor.invoke("How many events occurred in May from the event data?")
    return {"answer_md": str(resp["output"])}
