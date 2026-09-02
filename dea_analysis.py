import pandas as pd
import numpy as np
from scipy.optimize import linprog
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# STEP 1 - LOAD DATA
print("="*60)
print("STEP 1 - LOADING DATA")
print("="*60)

df = pd.read_csv('dmu_data_clean.csv', low_memory=False)
df = df.dropna(subset=['DMU_ID'])
df = df[df['DMU_ID'].str.startswith('DMU')].copy()
df = df.reset_index(drop=True)
df.columns = [
    'DMU_ID','Firm','Channel','Time_Period',
    'Spend','Sessions','Conversion_Rate','Revenue',
    'Source','Source_URL','Date_Accessed','Notes'
]
for col in ['Spend','Sessions','Conversion_Rate','Revenue']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"DMUs loaded: {len(df)}")
print(f"Firms: {df['Firm'].unique().tolist()}")
print(f"Channels: {df['Channel'].unique().tolist()}")

# STEP 2 - RUN BCC DEA MODEL
print()
print("="*60)
print("STEP 2 - RUNNING BCC INPUT-ORIENTED DEA")
print("="*60)

inputs = df[['Spend']].values.astype(float)
outputs = df[['Sessions','Conversion_Rate','Revenue']].values.astype(float)
inputs_norm = inputs / inputs.max(axis=0)
outputs_norm = outputs / outputs.max(axis=0)

def bcc_input_dea(inputs, outputs):
    n = len(inputs)
    m = inputs.shape[1]
    s = outputs.shape[1]
    scores = []
    for i in range(n):
        x0 = inputs[i]
        y0 = outputs[i]
        c = np.zeros(1 + n)
        c[0] = 1
        A_input = np.hstack([-x0.reshape(-1,1), inputs.T])
        b_input = np.zeros(m)
        A_output = np.hstack([np.zeros((s,1)), -outputs.T])
        b_output = -y0
        A_ub = np.vstack([A_input, A_output])
        b_ub = np.concatenate([b_input, b_output])
        A_eq = np.zeros((1, 1+n))
        A_eq[0,1:] = 1
        b_eq = np.array([1])
        bounds = [(0,None)] + [(0,None)]*n
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )
        scores.append(round(result.fun,4) if result.success else np.nan)
    return scores

scores = bcc_input_dea(inputs_norm, outputs_norm)
df['Efficiency_Score'] = scores
print("DEA complete.")

# STEP 3 - PRINT RESULTS
print()
print("="*60)
print("STEP 3 - EFFICIENCY SCORES")
print("="*60)
results = df[['DMU_ID','Firm','Channel','Time_Period','Efficiency_Score']]
print(results.sort_values('Efficiency_Score', ascending=False).to_string(index=False))

print()
print("="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Mean: {df['Efficiency_Score'].mean():.4f}")
print(f"Min: {df['Efficiency_Score'].min():.4f}")
print(f"Max: {df['Efficiency_Score'].max():.4f}")
print(f"Efficient DMUs (=1): {(df['Efficiency_Score']>=0.9999).sum()}")

print()
print("AVERAGE BY CHANNEL:")
print(df.groupby('Channel')['Efficiency_Score'].mean().sort_values(ascending=False).to_string())

print()
print("AVERAGE BY FIRM:")
print(df.groupby('Firm')['Efficiency_Score'].mean().sort_values(ascending=False).to_string())

print()
print("AVERAGE BY TIME PERIOD:")
print(df.groupby('Time_Period')['Efficiency_Score'].mean().to_string())

# STEP 4 - SLACK ANALYSIS
print()
print("="*60)
print("STEP 4 - SLACK ANALYSIS")
print("="*60)

input_slacks = (1 - df['Efficiency_Score'].values) * inputs[:,0]
slack_df = pd.DataFrame({
    'DMU_ID': df['DMU_ID'],
    'Firm': df['Firm'],
    'Channel': df['Channel'],
    'Time_Period': df['Time_Period'],
    'Efficiency_Score': df['Efficiency_Score'],
    'Input_Slack_Spend_GBP': input_slacks.round(0)
})
inefficient = slack_df[slack_df['Efficiency_Score'] < 0.9999]
print("Inefficient DMUs and input excess (£):")
print(inefficient.sort_values('Efficiency_Score').to_string(index=False))

# STEP 5 - SECOND STAGE REGRESSION
print()
print("="*60)
print("STEP 5 - SECOND STAGE REGRESSION (OLS)")
print("="*60)

reg_df = df.copy()
reg_df['Channel_clean'] = reg_df['Channel'].str.strip().str.upper()
reg_df['D_PAID_SEARCH'] = (reg_df['Channel_clean']=='PAID_SEARCH').astype(int)
reg_df['D_EMAIL'] = (reg_df['Channel_clean']=='EMAIL').astype(int)
reg_df['D_SOCIAL'] = (reg_df['Channel_clean'].isin(['SOCIAL_MEDIA','SOCIA_MEDIA'])).astype(int)
reg_df['D_Boohoo'] = (reg_df['Firm']=='Boohoo').astype(int)
reg_df['D_Next'] = (reg_df['Firm']=='Next').astype(int)
reg_df['D_THG'] = (reg_df['Firm']=='THG').astype(int)
reg_df['D_MandS'] = (reg_df['Firm']=='M&S').astype(int)
reg_df['D_FY2025'] = (reg_df['Time_Period']=='FY2025').astype(int)
reg_df['Log_Spend'] = np.log(reg_df['Spend'])

X_vars = [
    'D_PAID_SEARCH','D_EMAIL','D_SOCIAL',
    'D_Boohoo','D_Next','D_THG','D_MandS',
    'D_FY2025','Log_Spend'
]
X = sm.add_constant(reg_df[X_vars])
y = reg_df['Efficiency_Score']

ols = sm.OLS(y, X).fit()
print(ols.summary())

# STEP 6 - SAVE RESULTS
print()
print("="*60)
print("STEP 6 - SAVING RESULTS")
print("="*60)

df[['DMU_ID','Firm','Channel','Time_Period',
    'Spend','Sessions','Conversion_Rate',
    'Revenue','Efficiency_Score']].to_csv('dea_efficiency_scores.csv', index=False)

slack_df.to_csv('dea_slack_analysis.csv', index=False)

with open('regression_results.txt','w') as f:
    f.write(ols.summary().as_text())

print("Saved:")
print("  dea_efficiency_scores.csv")
print("  dea_slack_analysis.csv")
print("  regression_results.txt")
print()
print("ANALYSIS COMPLETE")
