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
    STOCK_XLSX = os.path.join(DATA_DIR, "stock_prices.xlsx")
    # switch to a file-backed SQLite DB
    DATABASE_PATH = os.path.join(DATA_DIR, "permits_data.sqlite3")
    PERMIT_SHEET = 0
    STOCK_SHEET = 0

    # Read Excel and normalize column names
    df = pd.read_excel(PERMIT_XLSX, sheet_name=PERMIT_SHEET)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df_stock = pd.read_excel(STOCK_XLSX, sheet_name=STOCK_SHEET)
    df_stock.columns = [col.lower().replace(" ", "_") for col in df_stock.columns]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')
        print(df['date'].head())
    # Use SQLAlchemy + SQLite: create/replace table on disk
    engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
    df.to_sql("event_data", con=engine, if_exists="replace", index=False)
    df_stock.to_sql("stock_data", con=engine, if_exists="replace", index=False)
    engine.dispose()

    # llm (unchanged)
    llm = ChatOllama(base_url="http://127.0.0.1:11434", model="llama3:8b", temperature=0, max_tokens=1000)

    # Connect LangChain to the file-backed SQLite via SQLAlchemy URI
    db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}")
    table_schema = db.get_table_info(table_names=["event_data"])  # or build a string manually
    result = db.run("SELECT COUNT(*) FROM event_data WHERE date LIKE '2024-06-%';")
    # print("Events in June:", result)
    prompt = f"Schema: {table_schema}\nQuestion: How many events occurred in June? SQL queries return results as lists of tuples. For example, [(5860,)] means the result is 5860."

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
    return {"answer_md": str(resp)}
