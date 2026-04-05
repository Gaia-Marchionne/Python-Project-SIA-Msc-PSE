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
└── requirements.txt
```
---
## Installation
Before Starting to run the code, you need to install Python 3 environment
    **pip install -r requirements.txt**
If pip does not work you need to precise in the terminal:
    **python3 -m pip install requirements.txt**

---
## Data Sources
All data from **FAOSTAT** (Food and Agriculture Organization of the United Nations):
- **QCL:** Crops & Livestock — yield, area harvested, production
- **EF:** Fertilizers by Nutrient — nitrogen use per country
- **EP:** Pesticides Use — total pesticide application
https://www.fao.org/faostat/

