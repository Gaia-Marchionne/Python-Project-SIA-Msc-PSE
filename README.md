# Crop Yield Regression — EU27

An analysis on agricultural crop yield (tonnes/hectare) across the 27 EU member states using FAOSTAT data.

**Target:** `Yield_t_ha` crop yield in tonnes per hectare  
**Coverage:** 27 EU countries and wheat crop
**Model:** OLS regression on log-transformed yield

## Research Question

What drives differences in wheat yield across EU27 countries, and how much of the variance is explained by nitrogen use, harvested area, and pesticide use?
---
## Project Structure
```
Python Project/
├── Fao Eu Crop Analysis.ipynb   
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualization.py s
│   └── modelling.py 
├── tests/
│   ├── test_data_loader.py     
├── data/                           
├── requirements.txt
└── environment.yml
```
---
## Installation
Before starting to run the code, you need to set up the environment. You can choose one of the following methods:
Option 1: Conda 
```bash
conda env create -f environment.yml
conda activate fao-eu27
python -m ipykernel install --user --name fao-eu27 --display-name "Python (fao-eu27)"
```
 Option 2: Python 3 Pip environment
 ```bash
    pip install -r requirements.txt
```
If pip does not work you need to precise in the terminal:
 ```bash
    python3 -m pip install -r requirements.txt
```

---
## Data Sources
All data from **FAOSTAT** (Food and Agriculture Organization of the United Nations):
- **QCL:** Crops & Livestock — yield, area harvested, production
- **EF:** Fertilizers by Nutrient — nitrogen use per country
- **EP:** Pesticides Use — total pesticide application
https://www.fao.org/faostat/

