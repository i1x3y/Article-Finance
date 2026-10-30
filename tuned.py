"""Hiperparametre seçimi: her 12 ayda bir (DL ile aynı takvim), eğitim penceresinin son 12 örneği
sıralı doğrulama seti olarak kullanılarak küçük, önceden belirlenmiş ızgaradan seçilir.
Seçilen parametrelerle modeller her ay yeniden tahmin edilir. Kullanım: python tuned.py TS i0 i1"""
import sys, warnings, itertools; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from pipeline import dd, T, LB, NFEAT, SEED, windows
GRID={
 'RIDGE':(lambda p:Ridge(**p), {'alpha':[0.1,1,10,100]}),
 'SVR':(lambda p:SVR(gamma='scale',**p), {'C':[1,10,100],'epsilon':[0.01,0.1]}),
 'RF':(lambda p:RandomForestRegressor(300,random_state=SEED,n_jobs=-1,**p), {'min_samples_leaf':[1,3,5],'max_features':[0.3,1.0]}),
 'GBM':(lambda p:GradientBoostingRegressor(random_state=SEED,n_estimators=200,**p), {'max_depth':[2,3],'learning_rate':[0.05,0.1],'subsample':[0.8,1.0]}),
 'XGB':(lambda p:xgb.XGBRegressor(random_state=SEED,n_estimators=200,**p), {'max_depth':[2,3,5],'learning_rate':[0.05,0.1],'subsample':[0.8,1.0]}),
}
def prep(k):
    tr=dd.iloc[:k]; y=dd[T].values
    feats=tr.corr().abs()[T].sort_values(ascending=False).head(NFEAT).index.tolist()
    S=MinMaxScaler((-1,1)).fit(tr[feats]).transform(dd[feats].iloc[:k]); X=windows(S)
    sy=MinMaxScaler((-1,1)).fit(y[LB:k].reshape(-1,1))
    return X[:-1], sy.transform(y[LB:k].reshape(-1,1)).ravel(), X[-1:], sy
def select(k):
    Xtr,ytr,_,_=prep(k); best={}
    for nm,(mk,g) in GRID.items():
        sc=[]
        for vals in itertools.product(*g.values()):
            p=dict(zip(g.keys(),vals)); m=mk(p).fit(Xtr[:-12],ytr[:-12])
            sc.append((np.mean((m.predict(Xtr[-12:])-ytr[-12:])**2),p))
        best[nm]=min(sc,key=lambda s:s[0])[1]
    return best
if __name__=='__main__':
    ts,i0,i1=map(int,sys.argv[1:]); n=len(dd); start=n-ts
    rows=[]; params=None
    for k in range(start+i0,start+i1):
        if (k-start)%12==0 or params is None: params=select(k-((k-start)%12)); print('params',k-start,params,flush=True)
        Xtr,ytr,xte,sy=prep(k); r={'idx':k}
        for nm,(mk,_) in GRID.items():
            r[nm+'_T']=sy.inverse_transform(mk(params[nm]).fit(Xtr,ytr).predict(xte).reshape(-1,1))[0,0]
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'results/tuned_{ts}_{i0}.csv',index=False); print('ok',ts,i0,i1)
