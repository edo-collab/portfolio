# Eurozone Macroeconomic Analysis (2000-2023)

An analysis of GDP growth, unemployment, and inflation across five European economies (Italy, Germany, France, Spain, Greece), with a focus on the 2008-2013 period covering both the global financial crisis (2008-2009) and the European sovereign debt crisis (2010-2013).

## Data Source

Data is sourced from the [World Bank Open Data](https://data.worldbank.org/) API, accessed via the `pandas_datareader` library.

**Indicators used:**
- GDP growth (annual %) — `NY.GDP.MKTP.KD.ZG`
- Unemployment rate (% of total labor force) — `SL.UEM.TOTL.ZS`
- Inflation, consumer prices (annual %) — `FP.CPI.TOTL.ZG`

**Countries:** Italy, Germany, France, Spain, Greece
**Time period:** 2000-2023

## Methodology

1. Data for each indicator was downloaded from the World Bank API for the five selected countries (2000-2023).
2. Data was reshaped from long format (country/year multi-index) to wide format (years as rows, countries as columns) for visualization.
3. Time series were plotted for each indicator to visualize trends across countries.
4. Descriptive statistics (mean, standard deviation, min, max) were computed for the full period and for the 2008-2013 crisis period.

## How to Run

1. Clone this repository.
2. Install the required libraries:
   ```
   pip install pandas matplotlib pandas_datareader
   ```
3. Open `analysis.ipynb` (in this folder) in Jupyter Notebook or VS Code.
4. Run all cells in order.

## Key Findings

- **Greece shows extreme volatility across all indicators**: the highest unemployment rate in the dataset (27.7% in 2013) alongside the deepest GDP contraction, consistently diverging from the other four countries during the 2008-2013 crisis.
- **Sharp divergence between Germany and Southern Europe during the crisis**: Germany's unemployment remained low and stable (mean 6.5%, std 1.1) between 2008-2013, while Spain's unemployment averaged 20.2% with much higher volatility (std 5.3) — reflecting the asymmetric impact of the crisis across the Eurozone.
- **Deflation episode (2013-2016)**: several countries, notably Greece and Spain, experienced negative inflation rates in this period, reflecting weak demand following the sovereign debt crisis.
- **Post-COVID rebound**: the 2020 GDP contraction was the sharpest of the entire 2000-2023 period, but the 2021 recovery was equally rapid — contrasting with the slow, uneven recovery seen after 2008-2013.
