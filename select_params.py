import sys, json, warnings; warnings.filterwarnings('ignore')
from tuned import select
from pipeline import dd
n=len(dd); out={}
for off in map(int,sys.argv[1:]):
    k=n-off; out[str(dd.index[k].strftime('%Y-%m'))]=select(k); print(off,'ok',flush=True)
json.dump(out,open(f"results/params_{'_'.join(sys.argv[1:])}.json",'w'),indent=1)
