from pathlib import Path
import sys,json, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from target_spin_gap import gap as target_gap, weak_coupling_fit
TWOPI=2*np.pi
root=Path(__file__).resolve().parents[1]; res=root/'results'
fs=np.concatenate([[0],np.logspace(-5,1,110)]); Js=TWOPI*fs*1e6
gaps=np.array([target_gap(J_rad_s=x) for x in Js]); g0,c=weak_coupling_fit(Js,gaps)
pd.DataFrame({'J_over_2pi_MHz':fs,'gap_s-1':gaps}).to_csv(res/'target_gap_scan.csv',index=False)
(res/'target_gap_fit.json').write_text(json.dumps({'gap_at_J0_s-1':float(gaps[0]),'gap0_fit_s-1':float(g0),'quadratic_coefficient_s':float(c)},indent=2))
print(gaps[0],g0,c)
