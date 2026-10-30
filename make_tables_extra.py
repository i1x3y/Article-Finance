"""Appendix tables: MASE / directional accuracy (Table 7) and selected hyperparameters (Table 8)."""
import json, glob, pandas as pd
LAB={'RW':'Random walk','AR':'AR($p$)','ARIMA':'ARIMA(2,1,2)','HOLT':'Holt (damped)','VAR':'VAR','RIDGE_T':'Ridge','SVR_T':'SVR','RF_T':'Random forest','GBM_T':'GBM','XGB_T':'XGBoost','ARIMA-XGB':'ARIMA--XGBoost','LSTM':'LSTM','GRU':'GRU','CNN':'CNN','CNN-LSTM':'CNN--LSTM','N-BEATS':'N-BEATS'}
R=pd.read_csv('results/mase_diracc.csv'); L=[]
for m in LAB:
    x=R[R.Model==m].set_index('Window')
    da=['--' if pd.isna(x.loc[w,'DirAcc']) else f"{x.loc[w,'DirAcc']:.2f}"+('$^{*}$' if x.loc[w,'p_binom']<0.05 else '') for w in [12,36,60]]
    L.append(LAB[m]+' & '+' & '.join(f"{x.loc[w,'MASE']:.2f}" for w in [12,36,60])+' & '+' & '.join(da)+r'\\')
open('paper/table_mase_diracc_body.tex','w').write('\n'.join(L))
P={}
for f in sorted(glob.glob('results/params_*.json')): P.update(json.load(open(f)))
pd.DataFrame({d:{m:str(v) for m,v in P[d].items()} for d in sorted(P)}).to_csv('results/selected_hyperparameters.csv')
print('ok')
