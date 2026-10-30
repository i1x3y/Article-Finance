"""DL modelleri – aynı rolling hat. Maliyet nedeniyle model 12 ayda bir yeniden eğitilir;
aradaki aylarda parametreler sabit, girdiler her ay güncel veriyle oluşturulur.
Her model 5 tohumla eğitilir, tahmin tohum ortalamasıdır (tek tohum sonuçları da saklanır).
Erken durdurma: eğitim penceresinin son 12 örneği sıralı doğrulama seti."""
import os, sys, warnings, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import MinMaxScaler
from pipeline import dd, lev, prev_all, T, LB, NFEAT

SEEDS=[0,1,2,3,4]; REFIT=12; NB_LB=24

def seed_all(s):
    os.environ['PYTHONHASHSEED']=str(s); random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def build(name, shape):
    inp=layers.Input(shape)
    if name=='LSTM': x=layers.LSTM(32)(inp); x=layers.Dropout(0.2)(x)
    elif name=='GRU': x=layers.GRU(32)(inp); x=layers.Dropout(0.2)(x)
    elif name=='CNN':
        x=layers.Conv1D(32,3,activation='relu',padding='causal')(inp); x=layers.MaxPooling1D(2)(x)
        x=layers.Flatten()(x); x=layers.Dense(32,activation='relu')(x)
    elif name=='CNN-LSTM':
        x=layers.Conv1D(32,3,activation='relu',padding='causal')(inp); x=layers.MaxPooling1D(2)(x); x=layers.LSTM(32)(x)
    elif name=='N-BEATS':   # generic N-BEATS (Oreshkin vd. 2020): backcast/forecast + residual bloklar, tek değişkenli
        r=layers.Flatten()(inp); fc=None
        for b in range(3):
            h=r
            for _ in range(4): h=layers.Dense(64,activation='relu')(h)
            back=layers.Dense(shape[0])(h); f=layers.Dense(1)(h)
            r=layers.Subtract()([r,back]); fc=f if fc is None else layers.Add()([fc,f])
        m=Model(inp,fc); m.compile('adam','mse'); return m
    out=layers.Dense(1)(x); m=Model(inp,out); m.compile('adam','mse'); return m

def fit_predict(name, Xtr, ytr, Xte_list, seed):
    seed_all(seed); m=build(name, Xtr.shape[1:])
    es=callbacks.EarlyStopping(patience=15,restore_best_weights=True)
    m.fit(Xtr[:-12],ytr[:-12],validation_data=(Xtr[-12:],ytr[-12:]),epochs=300,batch_size=16,
          shuffle=False,verbose=0,callbacks=[es])
    return m.predict(np.concatenate(Xte_list),verbose=0).ravel()

def run(ts, names):
    n=len(dd); start=n-ts; y_d=dd[T].values
    out={f'{nm}_s{s}':np.zeros(ts) for nm in names for s in SEEDS}
    for r0 in range(start,n,REFIT):
        r1=min(r0+REFIT,n); tr=dd.iloc[:r0]
        feats=tr.corr().abs()[T].sort_values(ascending=False).head(NFEAT).index.tolist()
        sx=MinMaxScaler((-1,1)).fit(tr[feats]); S=sx.transform(dd[feats])
        sy=MinMaxScaler((-1,1)).fit(y_d[LB:r0].reshape(-1,1))
        ys=sy.transform(y_d.reshape(-1,1)).ravel()
        Xtr=np.stack([S[j:j+LB] for j in range(0,r0-LB)]); ytr=ys[LB:r0]
        Xte=[S[k-LB:k][None] for k in range(r0,r1)]
        # N-BEATS: yalnızca hedefin geçmiş farkları, pencere 24
        u=ys.reshape(-1,1)
        Ntr=np.stack([u[j:j+NB_LB] for j in range(0,r0-NB_LB)]); Nytr=ys[NB_LB:r0]
        Nte=[u[k-NB_LB:k][None] for k in range(r0,r1)]
        for nm in names:
            for s in SEEDS:
                p=fit_predict(nm, Ntr if nm=='N-BEATS' else Xtr, Nytr if nm=='N-BEATS' else ytr,
                              Nte if nm=='N-BEATS' else Xte, s)
                out[f'{nm}_s{s}'][r0-start:r1-start]=sy.inverse_transform(p.reshape(-1,1)).ravel()
        print(f'  {ts}: refit {r0-start} bitti',flush=True)
    prev=prev_all.iloc[start:].values
    D=pd.DataFrame({k:prev+v for k,v in out.items()},index=lev.index[start:])
    for nm in names: D[nm]=D[[f'{nm}_s{s}' for s in SEEDS]].mean(axis=1)
    D.to_csv(f'results/dl_{ts}.csv'); return D

if __name__=='__main__':
    names=['LSTM','GRU','CNN','CNN-LSTM','N-BEATS']
    for ts in [int(a) for a in sys.argv[1:]]:
        run(ts,names); print('DONE',ts,flush=True)
