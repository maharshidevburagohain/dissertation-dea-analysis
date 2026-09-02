Digital Marketing Channel Efficiency — DEA Analysis
MSc Business Analytics Dissertation
University of Bristol Business School
Student: Maharshi Dev Buragohain
Student Number: 2742299
Supervisor: Marios Kremantzis
Repository Contents:
dmu_data_clean.csv — Final dataset of 40 DMUs across 5 UK e-commerce firms (ASOS, Boohoo, Next, THG, M&S), 4 digital marketing channels, and 2 financial years (FY2024 and FY2025)
dea_analysis.py — Python script implementing BCC input-oriented DEA model using SciPy linprog with HiGHS solver and second-stage OLS regression using statsmodels
dea_efficiency_scores.csv — Full efficiency scores for all 40 DMUs
dea_slack_analysis.csv — Input slack analysis showing excess spend per inefficient DMU
regression_results.txt — Full OLS regression output from second-stage analysis
How To Run The Analysis:
Requirements: Python 3.13, pandas, numpy, scipy, statsmodels
Run: python3 dea_analysis.py # dissertation-dea-analysis
