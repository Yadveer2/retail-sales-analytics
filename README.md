# Retail Sales Analytics

This project analyzes the Brazilian Olist e-commerce dataset and turns it into business-ready insights using SQL, Python, and Power BI.

It covers data preparation, exploratory analysis, customer segmentation, and time-series forecasting for revenue planning.

## What is included

- Cleaned and enriched sales data from Olist
- SQL-based KPI queries and revenue analysis
- Customer RFM segmentation using Python
- Revenue forecasting with Prophet
- Power BI dashboard for executive reporting

## Project structure

```text
retail-sales-analytics/
├── data/
│   ├── master.csv
│   └── raw Olist CSV files
├── docs/
│   ├── forecast/
│   ├── segmentation/
│   └── sql_results/
├── python/
│   ├── 05_rfm_segmentation.py
│   └── 06_forecasting.py
├── powerbi/
│   └── dashboard.pbix
├── sql/
│   └── olist.db
├── README.md
├── requirements.txt
└── .gitignore (optional, if added later)
```

## Data source

The project uses Olist e-commerce data, with a curated master dataset stored in `data/master.csv`.

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the analysis

### 1) Customer segmentation

```bash
python python/05_rfm_segmentation.py
```

This produces segment summaries and charts in `docs/segmentation/`.

### 2) Revenue forecasting

```bash
python python/06_forecasting.py
```

This creates a 6-month revenue forecast and saves results in `docs/forecast/`.

## Outputs

### SQL and business metrics

The project generates CSV files in `docs/sql_results/` with results such as:

- total revenue
- monthly revenue
- top categories
- top states
- top customers
- average order value
- year-over-year growth
- repeat customers
- quarterly revenue
- ABC classification

### Segmentation outputs

Files created in `docs/segmentation/` include:

- `rfm_segments.csv`
- `cluster_summary.csv`
- segment charts and visualizations

### Forecast outputs

Files created in `docs/forecast/` include:

- `forecast_6_months.csv`
- `forecast_chart.png`
- `forecast_components.png`

## Notes

- The Python scripts use relative paths based on the repository folder, so they work from the project root without hard-coded local machine paths.
- The project is designed to be easy to extend for additional revenue, retention, or product performance analysis.

## Technologies used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Facebook Prophet
- SQL
- Power BI

## License

This project is for educational and analytical purposes.
