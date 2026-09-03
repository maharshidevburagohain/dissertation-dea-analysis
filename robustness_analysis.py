"""
Robustness Analysis — Excluding M&S
Maharshi Dev Buragohain - MSc Business Analytics - University of Bristol
Re-runs BCC DEA model on 32 DMUs excluding M&S observations
to test sensitivity of main findings to M&S output measurement difference
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('dmu_data_clean.csv', low_memory=False)
df = df.dropna(subset=['DMU_ID'])
df = df[df['DMU_ID'].str.startswith('DMU')]
df.columns = ['DMU_ID','Firm','Channel','Time_Period','Spend','Sessions',
              'Conversion_Rate','Revenue','Source','Source_URL','Date_Accessed','Notes']
for col in ['Spend','Sessions','Conversion_Rate','Revenue']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Exclude M&S
df_no_mands = df[df['Firm'] != 'M&S'].copy().reset_index(drop=True)
print(f'DMUs in robustness check: {len(df_no_mands)}')

# Normalise
inputs = df_no_mands[['Spend']].values.astype(float)
outputs = df_no_mands[['Sessions','Conversion_Rate','Revenue']].values.astype(float)
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
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                        A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        scores.append(round(result.fun,4) if result.success else np.nan)
    return scores

scores = bcc_input_dea(inputs_norm, outputs_norm)
df_no_mands['Efficiency_Score'] = scores

print()
print('='*60)
print('ROBUSTNESS CHECK RESULTS - WITHOUT M&S')
print('='*60)
print(df_no_mands[['DMU_ID','Firm','Channel',
                    'Time_Period','Efficiency_Score']].to_string(index=False))
print()
print(f'Mean: {df_no_mands["Efficiency_Score"].mean():.4f}')
print()
print('BY CHANNEL:')
print(df_no_mands.groupby('Channel')['Efficiency_Score'].mean().sort_values(ascending=False))
print()
print('BY FIRM:')
print(df_no_mands.groupby('Firm')['Efficiency_Score'].mean().sort_values(ascending=False))

# Save results
df_no_mands[['DMU_ID','Firm','Channel','Time_Period',
             'Efficiency_Score']].to_csv('robustness_scores.csv', index=False)
print()
print('Results saved to robustness_scores.csv')

