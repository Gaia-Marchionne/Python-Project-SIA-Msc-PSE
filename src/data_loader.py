"""
data_loader.py
--------------
Loads and processes crop yield data from FAOSTAT (Food and Agriculture Organization).
The three FAOSTAT datasets used are:
  QCL — crop yield, area harvested, production
  EF  — nitrogen fertilizer use per country-year
  EP  — pesticide use per country-year

I decided to focus my research on the 27 EU member states and wheat (1990-2022).
Each ZIP contains one large CSV covering all countries and years.
I download once and reload from local memory if already downloaded.
"""
import os
import io
import zipfile
from typing import Tuple, List

import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# SETTINGS
""" I set the Target of my study as the column "Yield_t_ha" (crop yield in tonnes per hectare),this is the variable I want to predict and understand. 
The features will be the input variables that I use to predict the target, such as "AreaHarvested_ha", "NitrogenUse_kg_ha", and "PesticideUse_t".
I also define some constants for filtering the data, such as the list of EU27 countries and keywords to identify regional aggregates in the FAO data.
"""
TARGET = "Yield_t_ha"  # the column we want to predict
FOCUS_CROPS = ["Wheat"]
# All 27 EU member states
EU27_COUNTRIES = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands (Kingdom of the)",
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden",
]
# Keywords used to remove regional aggregates from the FAO data
REGION_KEYWORDS = [
    "World", "Africa", "Americas", "Asia", "Europe", "Oceania",
    "income", "developed", "developing", "region", "Total",
    "Low", "Middle", "High", "Eastern", "Western", "Southern",
    "Northern", "Central", "Net Food", "OECD",
]
# Links to the FAOSTAT bulk ZIP files
QCL_URL  = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
FERT_URL = "https://bulks-faostat.fao.org/production/Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip" 
PEST_URL = "https://bulks-faostat.fao.org/production/Inputs_Pesticides_Use_E_All_Data_(Normalized).zip" 

"""
Here I check if the "Area" name contains any of the region keywords, 
if it does, it's not a real country and we exclude it from our analysis.   
"""

def is_a_country(name: str) -> bool:

    return not any(kw.lower() in name.lower() for kw in REGION_KEYWORDS)

"""
Function to download and dezip CSV froma given URL
The resulting DataFrame is returned for further processing.
"""
def _download_zip_to_df(url: str, label: str) -> pd.DataFrame:
    print(f"  Downloading {label}...")
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_file = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(csv_file) as f:
            return pd.read_csv(f)


"""
Download the three FAO datasets (QCL, EF, EP)
The QCL dataset is processed to keep only the relevant elements (Yield, Area harvested, Production) and merged into a single wide table.
The EF and EP datasets are filtered to keep only the relevant rows (nitrogen fertilizer use per area and pesticide use) and aggregated by country-year.
"""
def download_fao_data(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)

    qcl_path  = os.path.join(data_dir, "fao_qcl.csv")
    fert_path = os.path.join(data_dir, "fao_fertilizer.csv")
    pest_path = os.path.join(data_dir, "fao_pesticides.csv")
    # For each dataset, check if the processed CSV already exists 
    # If not, download the ZIP, extract the relevant data, process it, and save it
    if not os.path.exists(qcl_path):
        raw = _download_zip_to_df(QCL_URL, "QCL (Crop yields)")

        # Keep only real countries — remove regional aggregates
        raw = raw[raw["Area"].apply(is_a_country)]

        # The raw data has one row per "element"
        # We extract each element separately and merge them into one wide table
        yield_df = (raw[raw["Element"] == "Yield"]
                    [["Area", "Item", "Year", "Value"]]
                    .rename(columns={"Value": "Yield_hg_ha"}))

        area_df  = (raw[raw["Element"] == "Area harvested"]
                    [["Area", "Item", "Year", "Value"]]
                    .rename(columns={"Value": "AreaHarvested_ha"}))

        prod_df  = (raw[raw["Element"] == "Production"]
                    [["Area", "Item", "Year", "Value"]]
                    .rename(columns={"Value": "Production_tonnes"}))

        df = yield_df.merge(area_df, on=["Area", "Item", "Year"], how="outer")
        df = df.merge(prod_df,       on=["Area", "Item", "Year"], how="outer")
        df.to_csv(qcl_path, index=False)
        print(f"  Saved: {qcl_path}")

    if not os.path.exists(fert_path):
        raw = _download_zip_to_df(FERT_URL, "EF (Fertilizers)")
        raw = raw[raw["Area"].apply(is_a_country)]
        df = (raw[
                (raw["Element"] == "Use per area of cropland") &
                raw["Item"].str.contains("nitrogen", case=False, na=False) &
                (raw["Unit"] == "kg/ha")
              ][["Area", "Year", "Value"]]
              .rename(columns={"Value": "NitrogenUse_kg_ha"})
              .groupby(["Area", "Year"], as_index=False)["NitrogenUse_kg_ha"].mean())

        df.to_csv(fert_path, index=False)
        print(f"  Saved: {fert_path}")

    if not os.path.exists(pest_path):
        raw = _download_zip_to_df(PEST_URL, "EP (Pesticides)")
        raw = raw[raw["Area"].apply(is_a_country)]

        df = (raw[raw["Element"].str.contains("Use", case=False, na=False)]
              [["Area", "Year", "Value"]]
              .rename(columns={"Value": "PesticideUse_t"})
              .groupby(["Area", "Year"], as_index=False)["PesticideUse_t"].sum())

        df.to_csv(pest_path, index=False)
        print(f"  Final data save in: {pest_path}")



"""
From the three raw DataFrames (QCL, EF, EP), we create a single processed DataFrame ready for modelling.
    Processing steps:
    - Merge datasets with year and country
    - Filter to EU27 countries and selected dynamic year range
    - Filter to wheat crops (cf. justification in the notebook)
    - Convert yield from hg/ha to t/ha
    - Remove zero, missing, and implausible yields (above 200 t/ha)
    - Fill missing nitrogen/pesticide with crop-year median
    - Add log-transforms for yield and all input variables
    - One-hot encode crop type — drop first category to avoid multicollinearity
"""
def process_fao_data(
    data_dir: str = "data",
    year_min: int = 1990,
    year_max: int = 2022,
    categorical_variable_list: list = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if categorical_variable_list is None:
        categorical_variable_list = ["Item"]

    qcl  = pd.read_csv(os.path.join(data_dir, "fao_qcl.csv"), low_memory=False)
    fert = pd.read_csv(os.path.join(data_dir, "fao_fertilizer.csv"))
    pest = pd.read_csv(os.path.join(data_dir, "fao_pesticides.csv"))

    # Merge all three on country and year
    df = (qcl
          .merge(fert, on=["Area", "Year"], how="left")
          .merge(pest, on=["Area", "Year"], how="left"))

    # Filter to EU27 and focus crops only
    df = df[df["Area"].isin(EU27_COUNTRIES)]
    df = df[df["Item"].isin(FOCUS_CROPS)]

    # Filter to the selected year range
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)]

    # Convert hg/ha → t/ha
    df["Yield_hg_ha"] = pd.to_numeric(df["Yield_hg_ha"], errors="coerce")
    df[TARGET] = df["Yield_hg_ha"] / 1_000

    # Remove rows with missing, zero, or implausible yield values
    df = df.dropna(subset=[TARGET, "AreaHarvested_ha"])
    df = df[(df[TARGET] > 0) & (df[TARGET] <= 200) & (df["AreaHarvested_ha"] > 0)]

    # Fill missing fertilizer and pesticide values with the crop-year median
    # Using 0 as a fill value would falsely imply that no inputs were used
    for col in ["NitrogenUse_kg_ha", "PesticideUse_t"]:
        group_median = df.groupby(["Item", "Year"])[col].transform("median")
        df[col] = df[col].fillna(group_median).fillna(df[col].median())

    # Log-transform inputs and target for the log-log OLS model.
    df["LOG_AreaHarvested"] = np.log(df["AreaHarvested_ha"])
    df["LOG_NitrogenUse"]   = np.log(df["NitrogenUse_kg_ha"])
    df["LOG_PesticideUse"]  = np.log(df["PesticideUse_t"])     
    df["LOG_Yield"]         = np.log(df[TARGET])

    # One-hot encode crop type
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.set_output(transform="pandas").fit_transform(df[categorical_variable_list])
    oh_column_names = list(encoded.columns)

    df = df.join(encoded)
    df = df.reset_index(drop=True)

    return df, oh_column_names


"""
Main function to get the processed FAO data ready for modelling.
The resulting DataFrame contains all relevant data from the 3 datasets downloaded
"""

def get_fao_data(
    data_dir: str = "data",
    year_min: int = 1990,
    year_max: int = 2023,
    categorical_variable_list: list = None,
) -> Tuple[pd.DataFrame, List[str]]:
    download_fao_data(data_dir)
    return process_fao_data(data_dir, year_min, year_max, categorical_variable_list)



"""
Data has to be split in two vectors: feature matrix X and a target vector y.
The function returns the feature matrix X and the target vector y, in order to be used for modelling and evaluation.
"""
def get_feature_target_split(
    df: pd.DataFrame,
    features: list = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
    features : list, optional

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
    """
    if features is None:
        features = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in (TARGET, "Yield_hg_ha", "Year")]
    return df[features].copy(), df[TARGET].copy()


"""
Helper function to load the full QCL dataset for EU27 countries without any crop filter.
it lets us visualise all crops and looking at the most relevant ones based on area, production, and yield
"""
def load_eu_raw(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the full QCL dataset for EU27 countries without any crop filter.

    This is used before selecting the focus crops — it lets us visualise
    all crops and looking at the most relevant for our study

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    pd.DataFrame
        Columns: Area, Item, Year, Yield_hg_ha, AreaHarvested_ha,
                 Production_tonnes, Yield_t_ha
    """
    qcl = pd.read_csv(os.path.join(data_dir, "fao_qcl.csv"), low_memory=False)

    # Filter to EU27 only without crop filter
    df = qcl[qcl["Area"].isin(EU27_COUNTRIES)].copy()

    # Convert yield to t/ha
    df["Yield_hg_ha"] = pd.to_numeric(df["Yield_hg_ha"])
    df[TARGET] = df["Yield_hg_ha"] / 1_000 

    # Remove zeros and missing
    df = df[(df[TARGET] > 0) & (df["AreaHarvested_ha"] > 0)]

    return df.reset_index(drop=True)