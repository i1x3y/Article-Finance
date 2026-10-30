import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import binomtest
from pipeline import lev, prev_all
MAIN=['ARIMA','HOLT','VAR','AR','RIDGE_T','SVR_T','RF_T','GBM_T','XGB_T','ARIMA-XGB','LSTM','GRU','CNN','CNN-LSTM','N-BEATS']
rows=[]
for ts in [12,36,60]:
    F=pd.read_csv(f'results/final_forecasts_{ts}.csv',index_col=0,parse_dates=True)
    y=F.ACTUAL.values; prev=prev_all.loc[F.index].values
    tr=lev.loc[:F.index[0]].iloc[:-1]; scale=np.abs(tr.diff().dropna()).mean()
    for m in ['RW']+MAIN:
        mae=np.abs(y-F[m]).mean()
        if m=='RW': da=np.nan; p=np.nan
        else:
            hit=np.sign(F[m].values-prev)==np.sign(y-prev); da=hit.mean(); p=binomtest(hit.sum(),len(hit),0.5,alternative='greater').pvalue
        rows.append([ts,m,mae/scale,da,p])
R=pd.DataFrame(rows,columns=['Window','Model','MASE','DirAcc','p_binom'])
R.to_csv('results/mase_diracc.csv',index=False)
print(R.pivot(index='Model',columns='Window',values=['MASE','DirAcc']).round(2).to_string())
