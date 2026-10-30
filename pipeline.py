"""Gerçek rolling (genişleyen pencere) değerlendirme hattı – DL hariç.
Her test ayında: özellik seçimi, ölçekleme ve TÜM modeller yalnızca o aya kadarki veriyle yeniden tahmin edilir.
Yayın gecikmesi: geç açıklanan seriler (sanayi, işgücü, konut, dış borç) bir ay ek gecikmeyle girer."""
import warnings, sys, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from scipy import stats

SEED=0; LB=12; NFEAT=21; T='tufe_yillik'
dr=pd.date_range('2012-01-01','2025-08-01',freq='MS')
def rd(f,pre=None):
    d=pd.read_csv(f'data/{f}')
    if pre: d.columns=[f'{pre}_{c}' for c in d.columns]
    d.index=dr[:len(d)]; return d
faiz=pd.read_csv('data/faiz_gecelik.txt',parse_dates=['tarih'],dayfirst=True).set_index('tarih')
fm=faiz.resample('MS').mean().reindex(dr).ffill(); fm.columns=[f'faiz_{c}' for c in fm.columns]
df=pd.concat([rd('tufe_yillik_aylik.txt'),rd('all.txt'),rd('sanayi_uretim.txt','sanayi'),
              rd('tuketici_guven_endeksi.txt','guven'),rd('ufe.txt','ufe'),fm],axis=1)
df=df.replace('-',np.nan).astype(float).ffill()          # yalnızca ileriye doldurma
# Tahmin anı: t-1 ayının TÜFE'si açıklandıktan hemen sonra (t ayının başı).
# Temel gecikme 1 ay (pencere t-12..t-1). Ek gecikmeler (yayın takvimine göre):
#   sanayi üretimi (m+2 ayının 10'u) ve kısa vadeli dış borç (m+2 ortası) -> son mevcut değer t-3 (+2 ek)
#   işgücü ve konut fiyat endeksi (m+1 içinde)                         -> son mevcut değer t-2 (+1 ek)
#   Not: dış borç hiçbir tahmin anında seçilmediği için bu değişiklik sonuçları etkilemez (kontrol edildi).
df=df.drop(columns=['isgucu_genc_kadin'])                # ilk 25 ayı eksik; örneklemi 2014'e kısaltıyordu
EXTRA={**{c:2 for c in df.columns if c.startswith(('sanayi_','dis_borc'))},
       **{c:1 for c in df.columns if c.startswith(('isgucu','konut'))}}
for c,l in EXTRA.items(): df[c]=df[c].shift(l)
df=df.dropna(axis=1,how='all')
lev=df[T]
dd=df.diff().iloc[1:]
dd=dd.dropna()                                           # ilk satırlar
lev=lev.loc[dd.index]                                    # hizalama
prev_all=df[T].shift(1).loc[dd.index]

def windows(S):
    return np.array([S[j:j+LB].ravel() for j in range(len(S)-LB+1)])  # satır j: [j..j+LB-1] → hedef j+LB

def run(ts,i0=None,i1=None):
    n=len(dd); start=n-ts
    P={k:[] for k in ['RW','AR','ARIMA','HOLT','VAR','RIDGE','SVR','RF','GBM','XGB','ARIMA-XGB']}
    y_d=dd[T].values
    rng=range(start,n) if i0 is None else range(start+i0,start+i1)
    for k in rng:
        tr=dd.iloc[:k]
        feats=tr.corr().abs()[T].sort_values(ascending=False).head(NFEAT).index.tolist()
        sx=MinMaxScaler((-1,1)).fit(tr[feats]); S=sx.transform(dd[feats].iloc[:k])
        Xall=windows(S)                       # son pencere: tahmin girdisi (satırlar k-LB..k-1)
        Xtr, xte = Xall[:-1], Xall[-1:]
        ytr=y_d[LB:k]
        sy=MinMaxScaler((-1,1)).fit(ytr.reshape(-1,1)); ytr_s=sy.transform(ytr.reshape(-1,1)).ravel()
        inv=lambda v: sy.inverse_transform(np.array(v).reshape(-1,1))[0,0]
        P['RW'].append(0.0)
        sel=ar_select_order(tr[T].values,maxlag=12,ic='aic')
        P['AR'].append(sel.model.fit().forecast(1)[0])
        ar=ARIMA(tr[T].values,order=(2,0,2)).fit(); a1=ar.forecast(1)[0]; P['ARIMA'].append(a1)
        hw=ExponentialSmoothing(lev.iloc[:k].values,trend='add',damped_trend=True,initialization_method='estimated').fit()
        P['HOLT'].append(hw.forecast(1)[0]-lev.iloc[k-1])
        vf=feats[:4] if T in feats[:4] else [T]+feats[:3]
        vm=VAR(tr[vf].values); lag=max(1,vm.select_order(maxlags=4).aic)
        vr=vm.fit(lag); P['VAR'].append(vr.forecast(tr[vf].values[-lag:],1)[0,vf.index(T)])
        for name,mod in [('RIDGE',Ridge(1.0)),('SVR',SVR(C=10,gamma='scale')),
                         ('RF',RandomForestRegressor(300,min_samples_leaf=3,random_state=SEED,n_jobs=-1)),
                         ('GBM',GradientBoostingRegressor(n_estimators=100,random_state=SEED)),
                         ('XGB',xgb.XGBRegressor(n_estimators=100,random_state=SEED))]:
            mod.fit(Xtr,ytr_s); P[name].append(inv(mod.predict(xte)))
        res=tr[T].values-ar.fittedvalues
        m=xgb.XGBRegressor(n_estimators=50,random_state=SEED).fit(Xtr,res[LB:k])
        P['ARIMA-XGB'].append(a1+m.predict(xte)[0])
    if i0 is not None:
        pd.DataFrame(P,index=lev.index[start+i0:start+i1]).to_csv(f'results/part_{ts}_{i0}.csv'); return None
    y=lev.iloc[start:].values; prev=prev_all.iloc[start:].values
    F=pd.DataFrame({k:prev+np.array(v) for k,v in P.items()},index=lev.index[start:]); F['ACTUAL']=y
    F.to_csv(f'results/forecasts_{ts}.csv'); return F

def dm_hln(e1,e2):
    d=e1**2-e2**2; n=len(d); s=d.mean()/np.sqrt(d.var(ddof=0)/n)*np.sqrt((n-1)/n)
    return s, 2*(1-stats.t.cdf(abs(s),n-1))

def evaluate(F,ts):
    from arch.bootstrap import MCS
    y=F['ACTUAL'].values; models=[c for c in F.columns if c!='ACTUAL']
    E={m:y-F[m].values for m in models}; e_rw=E['RW']
    rows=[]
    for m in models:
        e=E[m]; rmse=np.sqrt((e**2).mean()); mae=np.abs(e).mean()
        r2=1-(e**2).sum()/((y-y.mean())**2).sum()
        dm,p=(np.nan,np.nan) if m=='RW' else dm_hln(e,e_rw)
        rows.append([m,rmse,mae,rmse/np.sqrt((e_rw**2).mean()),r2,dm,p])
    R=pd.DataFrame(rows,columns=['Model','RMSE','MAE','RelRMSE_RW','R2','DM_vs_RW','p_vs_RW']).sort_values('RMSE')
    best=R.iloc[0].Model
    R['p_vs_best']=[np.nan if m==best else dm_hln(E[best],E[m])[1] for m in R.Model]
    L=pd.DataFrame({m:E[m]**2 for m in models})
    mcs=MCS(L,size=0.10,reps=2000,block_size=3,seed=SEED,method='R'); mcs.compute()
    R['MCS90']=R.Model.isin(mcs.included)
    R.to_csv(f'results/metrics_{ts}.csv',index=False)
    print(f'\n===== {ts} aylık değerlendirme penceresi =====')
    print(R.round(3).to_string(index=False))

def combine(ts):
    import glob
    D=pd.concat([pd.read_csv(f,index_col=0,parse_dates=True) for f in glob.glob(f'results/part_{ts}_*.csv')]).sort_index()
    F=D.add(prev_all.loc[D.index].values,axis=0); F['ACTUAL']=lev.loc[D.index].values
    F.to_csv(f'results/forecasts_{ts}.csv'); return F

if __name__=='__main__':
    if sys.argv[1]=='part': run(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
    elif sys.argv[1]=='combine': evaluate(combine(int(sys.argv[2])),int(sys.argv[2]))
    else:
        for ts in [int(a) for a in sys.argv[1:]]:
            evaluate(run(ts),ts)
