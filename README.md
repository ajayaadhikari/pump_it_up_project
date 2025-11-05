# Tanzanian Water Pumps: Predictive Maintenance Solution

## Project Overview

This project addresses the challenge faced by the Tanzanian Ministry of Water in ensuring functional water pumps for all citizens. Using provided and supplementary datasets, advanced data analysis, and machine learning, the project aims to predict which pumps are likely to fail, enabling proactive repair and replacement. The output helps guide a cost-effective, time-efficient triage strategy to maximize water access.

## Project Structure

```
.
├── data/                                # Contains input datasets (raw and processed CSV files)
├── requirements.txt                     # Python dependencies
├── src/                                 # Python scripts for the data science workflow
│   ├── load_data.py                     # Loads and inspects datasets
│   ├── preprocess.py                    # Data cleaning and feature engineering
│   ├── crossvalidation_hyperparameter_tuning.py # Model selection, cross-validation, and hyperparameter tuning
│   └── plot.py                          # Generates data visualizations
├── output/                              # Outputs such as predictions, cleaned data, and generated plots
├── README.md                            # Project overview, setup instructions, usage guide
└── .gitignore                           # Specifies files/folders to exclude from version control
```

## Getting Started

1. **Clone the Repository**
    ```
    git clone https://github.com/ajayaadhikari/pump_it_up_project.git
    cd pump_it_up_project
    ```

2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Download Datasets**
    - Download the datasets from [DrivenData, Pump It Up](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/)
    - Place the files inside the `data/` folder and change the data filenames in `load_data.py` accordingly.

4. **Workflow Steps**
    - load data: `load_data.py`
    - Clean and transform data: `python src/preprocess.py`
    - Model and tune: `python src/crossvalidation_hyperparameter_tuning.py`
    - Generate plots: `python src/plot.py`
    - Outputs (predictions, figures) appear in `output/`

