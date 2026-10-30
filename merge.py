import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pipeline import prev_all, evaluate
from dl import SEEDS
NM=['LSTM','GRU','CNN','CNN-LSTM','N-BEATS']
seedrows=[]
for ts in [12,36,60]:
    F=pd.read_csv(f'results/forecasts_{ts}.csv',index_col=0,parse_dates=True)
    import re
    parts={}
    for f in glob.glob(f'results/dlchunk_{ts}_*.csv'):
        i=int(re.search(rf'dlchunk_{ts}_(\d+)',f).group(1)); parts.setdefault(i,[]).append(pd.read_csv(f,index_col=0,parse_dates=True))
    C=pd.concat([pd.concat(parts[i],axis=1) for i in sorted(parts)])
    D=C.add(prev_all.loc[C.index].values,axis=0)
    for nm in NM: D[nm]=D[[f'{nm}_s{s}' for s in SEEDS]].mean(axis=1)
    assert len(D)==ts and (D.index==F.index).all()
    y=F['ACTUAL'].values; rw=np.sqrt(((y-F['RW'])**2).mean())
    for nm in NM:
        r=[np.sqrt(((y-D[f'{nm}_s{s}'])**2).mean())/rw for s in SEEDS]
        seedrows.append([ts,nm,np.mean(r),np.std(r),min(r),max(r)])
    A=pd.concat([F.drop(columns='ACTUAL'),D[NM],F[['ACTUAL']]],axis=1)
    A.to_csv(f'results/all_forecasts_{ts}.csv'); evaluate(A,ts)
S=pd.DataFrame(seedrows,columns=['Window','Model','RelRMSE_mean','RelRMSE_sd','min','max'])
S.to_csv('results/dl_seed_variability.csv',index=False)
print('\n=== Tohumlar arası değişkenlik (tek tohum Göreli RMSE) ==='); print(S.round(3).to_string(index=False))
