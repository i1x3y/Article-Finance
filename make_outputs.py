import re
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
MAIN=['RW','AR','ARIMA','HOLT','VAR','RIDGE_T','SVR_T','RF_T','GBM_T','XGB_T','ARIMA-XGB','LSTM','GRU','CNN','CNN-LSTM','N-BEATS']
LAB={'RW':'Random walk','AR':'AR($p$)','ARIMA':'ARIMA(2,1,2)','HOLT':'Holt (damped)','VAR':'VAR','RIDGE_T':'Ridge','SVR_T':'SVR',
     'RF_T':'Random forest','GBM_T':'GBM','XGB_T':'XGBoost','ARIMA-XGB':'ARIMA--XGBoost','LSTM':'LSTM','GRU':'GRU','CNN':'CNN','CNN-LSTM':'CNN--LSTM','N-BEATS':'N-BEATS'}
GRP={'RW':'Benchmarks','AR':'Benchmarks','ARIMA':'Econometric','HOLT':'Econometric','VAR':'Econometric','RIDGE_T':'Machine learning','SVR_T':'Machine learning',
     'RF_T':'Machine learning','GBM_T':'Machine learning','XGB_T':'Machine learning','ARIMA-XGB':'Hybrid','LSTM':'Deep learning','GRU':'Deep learning','CNN':'Deep learning','CNN-LSTM':'Deep learning','N-BEATS':'Deep learning'}
M={ts:pd.read_csv(f'results/metrics_{ts}.csv').set_index('Model') for ts in [12,36,60]}
def star(p): return '' if pd.isna(p) else ('$^{***}$' if p<.01 else '$^{**}$' if p<.05 else '$^{*}$' if p<.10 else '')
# --- Tablo: göreli RMSE + DM yıldızları + MCS
lines=[r'\begin{table}[!ht]\centering\small',
 r'\caption{Out-of-sample accuracy relative to the random walk}\label{tab:main}',
 r'\begin{tabular}{l rrr rrr}\toprule',
 r' & \multicolumn{3}{c}{Relative RMSE} & \multicolumn{3}{c}{RMSE (pp)}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
 r'Model & 12 m & 36 m & 60 m & 12 m & 36 m & 60 m\\\midrule']
g0=None
for m in MAIN:
    if GRP[m]!=g0: g0=GRP[m]; lines.append(r'\multicolumn{7}{l}{\textit{'+g0+r'}}\\')
    rel=[]; rm=[]
    for ts in [12,36,60]:
        r=M[ts].loc[m]; best=M[ts].RMSE.idxmin()
        s=f"{r.RelRMSE_RW:.3f}"+star(r.p_vs_RW)
        if r.MCS90: s=r'\underline{'+s+'}'
        if m==best: s=r'\textbf{'+s+'}'
        rel.append(s); rm.append(f"{r.RMSE:.2f}")
    lines.append(f"\\quad {LAB[m]} & "+' & '.join(rel)+' & '+' & '.join(rm)+r'\\')
lines+= [r'\bottomrule\end{tabular}',
 r'\begin{minipage}{0.95\linewidth}\footnotesize\vspace{2pt} Notes: Relative RMSE is the ratio of each model’s RMSE to that of the random walk; values below one indicate improvement. Asterisks denote rejection of equal predictive accuracy against the random walk in a two-sided Diebold--Mariano test with the Harvey--Leybourne--Newbold small-sample correction ($^{*}$10\%, $^{**}$5\%, $^{***}$1\%). Underlined models belong to the 90\% Model Confidence Set (range statistic, block bootstrap, 2{,}000 replications). Bold marks the lowest RMSE in each evaluation window. Machine learning hyperparameters are selected on a rolling validation sample (see Section~\ref{sec:tuning}).\end{minipage}',
 r'\end{table}']
open('paper/table_main.tex','w').write('\n'.join(lines))
# --- Alt dönem tablosu
S=pd.read_csv('results/subperiods_60.csv',index_col=0)
cols=['ARIMA','RIDGE_T','RF_T','GBM_T','XGB_T','ARIMA-XGB','GRU','LSTM','N-BEATS']
t=[r'\begin{table}[!ht]\centering\small\caption{Relative RMSE by sub-period (60-month evaluation window)}\label{tab:sub}',
   r'\begin{tabular}{l r'+'r'*len(cols)+r'}\toprule','Sub-period & RW RMSE & '+' & '.join(LAB[c] for c in cols)+r'\\\midrule']
for p,r in S.iterrows():
    vals=[r[c] for c in cols]; b=min(vals)
    t.append(re.sub(r'(\d{4}):0?(\d{1,2})',lambda m:f'{m.group(1)}:M{int(m.group(2))}',p).replace('–','--')+f' & {r["RW RMSE"]:.2f} & '+' & '.join((r'\textbf{%.2f}' if v==b else '%.2f')%v for v in vals)+r'\\')
t+= [r'\bottomrule\end{tabular}\end{table}']
open('paper/table_subperiods.tex','w').write('\n'.join(t))
# --- Şekil 1: 60 ay tahminler
F=pd.read_csv('results/final_forecasts_60.csv',index_col=0,parse_dates=True)
fig,ax=plt.subplots(2,1,figsize=(6.5,5.2),sharex=True)
ax[0].plot(F.index,F.ACTUAL,'k',lw=1.8,label='Actual')
for m,st in [('RW',':'),('RF_T','-'),('XGB_T','--'),('ARIMA-XGB','-.')]:
    ax[0].plot(F.index,F[m],st,lw=1,label=LAB[m].replace('--','–'))
ax[0].set_ylabel('Annual CPI inflation (%)'); ax[0].legend(frameon=False,ncol=3,fontsize=8)
for m,st in [('RW',':'),('RF_T','-'),('GRU','--')]:
    ax[1].plot(F.index,(F.ACTUAL-F[m]).abs().rolling(6).mean(),st,lw=1,label=LAB[m])
ax[1].set_ylabel('6-month rolling MAE (pp)'); ax[1].legend(frameon=False,ncol=3,fontsize=8)
fig.tight_layout(); fig.savefig('paper/fig_forecasts_60.pdf'); fig.savefig('paper/fig_forecasts_60.png',dpi=200)
# --- Şekil 2: göreli RMSE ısı haritası
H=pd.DataFrame({f'{ts} m':M[ts].loc[MAIN,'RelRMSE_RW'] for ts in [12,36,60]}).rename(index=lambda m:LAB[m].replace('$','').replace('--','–'))
fig,ax=plt.subplots(figsize=(4.2,5.5)); im=ax.imshow(H.values,cmap='RdBu_r',vmin=0.5,vmax=1.5,aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(H.columns); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(3): ax.text(j,i,f'{H.values[i,j]:.2f}',ha='center',va='center',fontsize=7)
fig.colorbar(im,ax=ax,shrink=0.6,label='RMSE relative to random walk'); fig.tight_layout()
fig.savefig('paper/fig_relrmse_heatmap.pdf'); fig.savefig('paper/fig_relrmse_heatmap.png',dpi=200)
UFE={'ufe':'headline','madencilik':'mining and quarrying','gida':'food products','icecek':'beverages','tutun':'tobacco products','tekstil':'textiles','giyim':'wearing apparel','petrol_urunleri':'coke and refined petroleum products','ana_metaller':'basic metals','motorlu_kara_tasit':'motor vehicles','ham_petrol_doğalgaz':'crude petroleum and natural gas','imalat':'manufacturing','rafine_petrol':'refined petroleum products'}
FX={'usd':'US dollar','eur':'Euro','gbp':'Pound sterling','jpy':'Japanese yen (per 100)'}
def nice(c):
    if c=='tufe_yillik': return 'Annual CPI inflation'
    if c=='tufe_aylik': return 'Monthly CPI inflation'
    if c.startswith('ufe_') and c[4:] in UFE: return 'PPI, '+UFE[c[4:]]
    k,_,t=c.partition('_')
    if k in FX and t in ('alis','satis'): return f"{FX[k]}/TRY {'buying' if t=='alis' else 'selling'} rate"
    if c=='usd': return 'Brent crude oil price'
    return c
# --- Şekil 3: değişken önemi
I=pd.read_csv('results/permutation_importance_60.csv',index_col=0).head(10)[::-1]
fig,ax=plt.subplots(figsize=(5.5,3.4)); y=np.arange(len(I))
ax.barh(y-0.2,I.XGB,0.4,label='XGBoost'); ax.barh(y+0.2,I.RF,0.4,label='Random forest')
ax.set_yticks(y); ax.set_yticklabels([nice(c) for c in I.index],fontsize=8); ax.set_xlabel('Increase in RMSE after permutation (pp)'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('paper/fig_importance.pdf'); fig.savefig('paper/fig_importance.png',dpi=200)
# --- Şekil 4: DL tohum değişkenliği
D=pd.read_csv('results/dl_seed_variability.csv')
fig,ax=plt.subplots(figsize=(6.5,2.8))
nm=['LSTM','GRU','CNN','CNN-LSTM','N-BEATS']; x=np.arange(len(nm))
for k,ts in enumerate([12,36,60]):
    d=D[D.Window==ts].set_index('Model').loc[nm]
    ax.errorbar(x+(k-1)*0.22,d.RelRMSE_mean,yerr=[d.RelRMSE_mean-d['min'],d['max']-d.RelRMSE_mean],fmt='o',ms=4,capsize=2,label=f'{ts} m')
ax.axhline(1,color='k',lw=0.8,ls=':'); ax.set_xticks(x); ax.set_xticklabels([n.replace('-','–') for n in nm])
ax.set_ylabel('Relative RMSE (single seed)'); ax.legend(frameon=False,ncol=3)
fig.tight_layout(); fig.savefig('paper/fig_dl_seeds.pdf'); fig.savefig('paper/fig_dl_seeds.png',dpi=200)
print(open('paper/table_main.tex').read()[:1500])
