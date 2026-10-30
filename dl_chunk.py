import sys, numpy as np, pandas as pd
import dl
from dl import *
def chunk(ts, idx, names=['LSTM','GRU','CNN','CNN-LSTM','N-BEATS'], tag=''):
    n=len(dd); start=n-ts; y_d=dd[T].values
    r0=start+idx*REFIT; r1=min(r0+REFIT,n); tr=dd.iloc[:r0]
    feats=tr.corr().abs()[T].sort_values(ascending=False).head(NFEAT).index.tolist()
    sx=MinMaxScaler((-1,1)).fit(tr[feats]); S=sx.transform(dd[feats])
    sy=MinMaxScaler((-1,1)).fit(y_d[LB:r0].reshape(-1,1)); ys=sy.transform(y_d.reshape(-1,1)).ravel()
    Xtr=np.stack([S[j:j+LB] for j in range(0,r0-LB)]); ytr=ys[LB:r0]; Xte=[S[k-LB:k][None] for k in range(r0,r1)]
    u=ys.reshape(-1,1); Ntr=np.stack([u[j:j+NB_LB] for j in range(0,r0-NB_LB)]); Nytr=ys[NB_LB:r0]; Nte=[u[k-NB_LB:k][None] for k in range(r0,r1)]
    out={}
    for nm in names:
        for s in SEEDS:
            nb=nm=='N-BEATS'
            p=fit_predict(nm, Ntr if nb else Xtr, Nytr if nb else ytr, Nte if nb else Xte, s)
            out[f'{nm}_s{s}']=sy.inverse_transform(p.reshape(-1,1)).ravel()
    pd.DataFrame(out,index=lev.index[r0:r1]).to_csv(f'results/dlchunk_{ts}_{idx}{tag}.csv'); print('ok',ts,idx)
G={'a':['LSTM','GRU','CNN'],'b':['CNN-LSTM','N-BEATS']}
if len(sys.argv)>3: chunk(int(sys.argv[1]),int(sys.argv[2]),G[sys.argv[3]],sys.argv[3])
else: chunk(int(sys.argv[1]),int(sys.argv[2]))
