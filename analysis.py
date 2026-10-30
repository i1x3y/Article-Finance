import warnings, glob, sys; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from pipeline import dd, df, lev, prev_all, T, LB, NFEAT, evaluate, windows
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.width',200)
part=sys.argv[1]
if part=='eval':
    for ts in [12,36,60]:
        A=pd.read_csv(f'results/all_forecasts_{ts}.csv',index_col=0,parse_dates=True)
        Tn=pd.concat([pd.read_csv(f) for f in glob.glob(f'results/tuned_{ts}_*.csv')]).sort_values('idx')
        prev=prev_all.values[Tn.idx.values]
        for c in [c for c in Tn if c.endswith('_T')]: A.insert(len(A.columns)-1,c,prev+Tn[c].values)
        A.to_csv(f'results/final_forecasts_{ts}.csv'); evaluate(A,ts)
if part=='sub':
    A=pd.read_csv('results/final_forecasts_60.csv',index_col=0,parse_dates=True)
    per={'2020:09–2021:12':('2020-09','2021-12'),'2022:01–2023:06':('2022-01','2023-06'),
         '2023:07–2024:05':('2023-07','2024-05'),'2024:06–2025:08':('2024-06','2025-08')}
    M=['RW','ARIMA','RIDGE_T','RF_T','GBM_T','XGB_T','ARIMA-XGB','GRU','LSTM','N-BEATS']
    out={}
    for p,(a,b) in per.items():
        s=A.loc[a:b]; rw=np.sqrt(((s.ACTUAL-s.RW)**2).mean())
        out[p]={m:np.sqrt(((s.ACTUAL-s[m])**2).mean())/rw for m in M}|{'RW RMSE':rw,'n':len(s)}
    R=pd.DataFrame(out).T; R.to_csv('results/subperiods_60.csv'); print(R.round(2).to_string())
if part=='desc':
    V={'tufe_yillik':'Annual CPI inflation','tufe_aylik':'Monthly CPI inflation'}
    D=df.loc[dd.index]
    rows=[]
    for c in D.columns:
        x=D[c].dropna()
        rows.append([c,x.mean(),x.std(),x.min(),x.max(),x.skew(),x.kurt(),adfuller(x,autolag='AIC')[1],adfuller(x.diff().dropna(),autolag='AIC')[1]])
    R=pd.DataFrame(rows,columns=['Variable','Mean','SD','Min','Max','Skew','Kurt','ADF_p_level','ADF_p_diff'])
    R.to_csv('results/descriptives.csv',index=False)
    print(len(R),'değişken'); print(R[R.Variable.isin(['tufe_yillik','tufe_aylik','faiz_borc_verme','ufe_genel'] + [c for c in R.Variable if c.startswith('usd')][:1])].round(3).to_string(index=False))
    print('Seviyede durağan olmayan (p>0.05):',(R.ADF_p_level>0.05).sum(),'| farkta durağan olmayan:',(R.ADF_p_diff>0.05).sum())
if part=='imp':
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from tuned import select, prep, GRID
    n=len(dd); k0=n-60; prm=select(k0)
    tr=dd.iloc[:k0]; feats=tr.corr().abs()[T].sort_values(ascending=False).head(NFEAT).index.tolist()
    sx=MinMaxScaler((-1,1)).fit(tr[feats]); S=sx.transform(dd[feats]); y=dd[T].values
    sy=MinMaxScaler((-1,1)).fit(y[LB:k0].reshape(-1,1))
    Xtr=np.array([S[j:j+LB].ravel() for j in range(k0-LB)]); ytr=sy.transform(y[LB:k0].reshape(-1,1)).ravel()
    Xte=np.array([S[k-LB:k].ravel() for k in range(k0,n)]); yte=y[k0:]
    rng=np.random.default_rng(0); res={}
    for nm in ['XGB','RF']:
        m=GRID[nm][0](prm[nm]).fit(Xtr,ytr)
        f=lambda X: sy.inverse_transform(m.predict(X).reshape(-1,1)).ravel()
        base=np.sqrt(((yte-f(Xte))**2).mean()); imp={}
        for i,c in enumerate(feats):
            cols=[l*NFEAT+i for l in range(LB)]; inc=[]
            for r in range(20):
                Xp=Xte.copy(); Xp[:,cols]=Xp[rng.permutation(len(Xp))][:,cols]
                inc.append(np.sqrt(((yte-f(Xp))**2).mean())-base)
            imp[c]=np.mean(inc)
        res[nm]=imp
    R=pd.DataFrame(res).sort_values('XGB',ascending=False); R.to_csv('results/permutation_importance_60.csv')
    print('Parametreler:',{k:prm[k] for k in ['XGB','RF']}); print(R.round(3).head(10).to_string())
