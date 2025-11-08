import os
import pandas as pd
import sqlite3
from typing import Optional
from utils import create_park_name_mapping

DIAMOND_FIELD_COLUMNS = ["Name of Field", 
                        "Name of Park Site", 
                        "Neighbourhood", 
                        "Site Address", 
                        "Diamonds: Dimension Home to Pitchers Plate - m", 
                        "Diamonds: Dimension Home to Second Base -m", 
                        "Diamonds: Dimension Home to Infield Arc - m", 
                        "Diamonds: Home to First Base Path - m"]
RECTANGULAR_FIELD_COLUMNS = ["Name of Field", 
                        "Name of Park Site", 
                        "Neighbourhood", 
                        "Site Address", 
                        "Rectangular Field Dimension: Length - m", 
                        "Rectangular Field Dimension: Width - m", 
                        "Rectangular Field Area From Length x Width - m²"]

class DataLayer:
    def __init__(self):
        """
        Initialize the DataLayer with the database path and data directory.
        """
        self.db_path = "data/parks_data.sqlite3"
        self.data_dir = "data"
        self.connection: Optional[sqlite3.Connection] = None

    def initialize_database(self) -> None:
        """
        Create a SQLite database connection and populate it with data from Excel files.
        """
        # Connect to SQLite database (file-based)
        self.connection = sqlite3.connect(self.db_path)

        # Read Excel data
        labor_xlsx = os.path.join(self.data_dir, "6 Mowing Reports to Jun 20 2025.xlsx")
        labor_df = pd.read_excel(labor_xlsx, sheet_name=0)

        field_size_path = os.path.join(self.data_dir, "3 vsfs_master_inventory_fieldsizes.xlsx")
        diamond_field_size_df_full = pd.read_excel(field_size_path, sheet_name=1).fillna("None")
        rectangular_field_size_df_full = pd.read_excel(field_size_path, sheet_name=2).fillna("None")
        event_df = pd.read_excel(os.path.join(self.data_dir, "4 Permits_2024.xlsx"), sheet_name=0)
        activity_type_df = pd.read_excel(os.path.join(self.data_dir, "6 Maint Activity Types- Mar 2025.xlsx"), sheet_name=0, skiprows=3, usecols='B,C,D')
        park_GIS_df = pd.read_excel(os.path.join(self.data_dir, "parks.xlsx"), sheet_name=0)
        order_df = pd.read_excel(os.path.join(self.data_dir, "6 Stanley order list.xlsx"), sheet_name=0)
        # Create park name mapping
        mapping_df = create_park_name_mapping(labor_df, park_GIS_df)
        
        # Normalize column names
        labor_df.columns = [str(c).strip() for c in labor_df.columns]
        diamond_field_size_df_full.columns = [str(c).strip() for c in diamond_field_size_df_full.columns]
        rectangular_field_size_df_full.columns = [str(c).strip() for c in rectangular_field_size_df_full.columns] 
        event_df.columns = [str(c).strip() for c in event_df.columns]
        activity_type_df.columns = [str(c).strip() for c in activity_type_df.columns]
        park_GIS_df.columns = [str(c).strip() for c in park_GIS_df.columns]
        order_df.columns = [str(c).strip() for c in order_df.columns]
        # Create truncated versions of field size dataframes
        diamond_field_size_df = diamond_field_size_df_full.filter(items=DIAMOND_FIELD_COLUMNS)
        rectangular_field_size_df = rectangular_field_size_df_full.filter(items=RECTANGULAR_FIELD_COLUMNS)

        # Convert columns to appropriate types
        if "Posting Date" in labor_df.columns:
            labor_df["Posting Date"] = pd.to_datetime(labor_df["Posting Date"], errors="coerce")
            labor_df = labor_df.dropna(subset=["Posting Date"])
        if 'Date' in event_df.columns:
            event_df['Date'] = pd.to_datetime(event_df['Date'], format='%b %d, %Y', errors='coerce')

        if "Val.in rep.cur." in labor_df.columns:
            labor_df["Val.in rep.cur."] = pd.to_numeric(labor_df["Val.in rep.cur."], errors="coerce").fillna(0.0)
        columns_to_convert = ["Diamonds: Dimension Home to Pitchers Plate - m", 
                            "Diamonds: Dimension Home to Second Base -m", 
                            "Diamonds: Dimension Home to Infield Arc - m", 
                            "Diamonds: Home to First Base Path - m",
                            "Rectangular Field Dimension: Length - m", 
                            "Rectangular Field Dimension: Width - m", 
                            "Rectangular Field Area From Length x Width - m²"]
        for col in columns_to_convert:
            if col in diamond_field_size_df.columns:
                diamond_field_size_df[col] = pd.to_numeric(diamond_field_size_df[col], errors="coerce")
            if col in rectangular_field_size_df.columns:
                rectangular_field_size_df[col] = pd.to_numeric(rectangular_field_size_df[col], errors="coerce")
        park_GIS_df["AREA_HA"] = pd.to_numeric(park_GIS_df["AREA_HA"], errors="coerce").fillna(0.0)
        
        # Rename column
        order_df = order_df.rename(columns={"Total act.costs": "Cost"})
        # Write data to SQLite tables
        labor_df.to_sql("labor_data", self.connection, if_exists="replace", index=False)
        diamond_field_size_df_full.to_sql("diamond_field_size_data_full", self.connection, if_exists="replace", index=False)
        rectangular_field_size_df_full.to_sql("rectangular_field_size_data_full", self.connection, if_exists="replace", index=False)
        event_df.to_sql("event_data", self.connection, if_exists="replace", index=False)
        activity_type_df.to_sql("activity_type_data", self.connection, if_exists="replace",index=False)
        diamond_field_size_df.to_sql("diamond_field_size_data", self.connection, if_exists="replace", index=False)
        rectangular_field_size_df.to_sql("rectangular_field_size_data", self.connection, if_exists="replace", index=False)
        park_GIS_df.to_sql("park_GIS_data", self.connection, if_exists="replace", index=False)
        order_df.to_sql("order_data", self.connection, if_exists="replace", index=False)
        mapping_df.to_sql("park_name_mapping", self.connection, if_exists="replace", index=False)

        return self
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        """
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        return self.connection

    def close_connection(self) -> None:
        """
        Close the SQLite database connection.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def show_tables_schemas(self) -> None:
        """
        Print the names and schemas of all tables in the database.
        """
        if self.connection is None:
            self.get_connection()
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables in the database:")
        for table in tables:
            table_name = table[0]
            print(f"- {table_name}")
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            schema = cursor.fetchall()
            print(f"  Schema for table '{table_name}':")
            for column in schema:
                print(f"    - {column[1]} ({column[2]})")  # column[1] = column name, column[2] = data type
