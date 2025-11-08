import pandas as pd
import re

def clean_park_name(co_object_name: str) -> str:
    """
    Extract first word after the first dash.
    If no dash, return first word of the string.
    E.g., "RFT- Alice Town Pk PTurf Mow/Maint" -> "Alice"
    E.g., "Alice Town Park" -> "Alice"
    """
    if pd.isna(co_object_name):
        return ""
    
    text = str(co_object_name).strip()
    
    # Split at first "-" if it exists
    if '-' in text:
        # Get text after first dash
        after_dash = text.split('-', 1)[1].strip()
    else:
        # No dash, use the whole text
        after_dash = text
    
    # Get first word
    words = after_dash.split()
    if not words:
        return ""
    
    return words[0]


def get_first_word(park_name: str) -> str:
    """Extract first word for fuzzy matching."""
    if pd.isna(park_name) or not park_name:
        return ""
    
    # Split and check if there are any words
    words = str(park_name).strip().split()
    
    if not words:  # Empty list after split
        return ""
    
    return words[0].lower()


def create_park_name_mapping(labor_df: pd.DataFrame, park_GIS_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping between dirty and clean park names.
    Returns a dataframe with: CO_Object_Name, clean_park_name, matched_PARKNAME, first_word
    """
    mapping_data = []
    
    # Get unique dirty park names
    unique_co_names = labor_df['CO Object Name'].unique()
    
    # Get clean park names with first words - create lookup dictionary
    park_lookup = {
        get_first_word(name): name 
        for name in park_GIS_df['PARKNAME'].unique()
    }
    
    for co_name in unique_co_names:
        clean_name = clean_park_name(co_name)
        first_word = get_first_word(clean_name)
        
        # Find matching park
        matched_park = park_lookup.get(first_word, None)
        
        mapping_data.append({
            'CO_Object_Name': co_name,
            'clean_park_name': clean_name,
            'matched_PARKNAME': matched_park,
            'first_word': first_word
        })
    
    return pd.DataFrame(mapping_data)