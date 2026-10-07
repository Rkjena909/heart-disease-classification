from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,f1_score
R=Path(__file__).resolve().parents[1]
raw=pd.read_csv(R/"data/heart_disease.csv")
summary=json.loads((R/"results/experiment_summary.json").read_text())
assert summary["source_sha256"]==hashlib.sha256((R/"data/heart_disease.csv").read_bytes()).hexdigest()
assign=pd.read_csv(R/"results/tables/split_assignments.csv")
train=set(assign.loc[assign.split.eq("train"),"id"]);test=set(assign.loc[assign.split.eq("holdout"),"id"])
assert train.isdisjoint(test) and train|test==set(raw.id)
assert len(train)==summary["train_records"] and len(test)==summary["holdout_records"]
assert not set(summary["predictors"]) & {"id","dataset","num"}
pred=pd.read_csv(R/"results/predictions/holdout_predictions.csv")
held=pd.read_csv(R/"results/tables/holdout_metrics.csv")
cv=pd.read_csv(R/"results/tables/cross_validation_metrics.csv")
assert summary["selected_model"]==cv.sort_values("cv_roc_auc",ascending=False).iloc[0].model
for row in held.itertuples():
    p=pred.loc[pred.model.eq(row.model)]
    assert len(p)==len(test) and set(p.id)==test and not p.id.duplicated().any()
    assert np.isfinite(p.score).all() and p.score.between(0,1).all()
    truth=raw.set_index("id").loc[p.id,"num"].gt(0).astype(int).to_numpy()
    assert np.array_equal(truth,p.target.to_numpy())
    assert np.array_equal(p.prediction.to_numpy(),(p.score.to_numpy()>=.5).astype(int))
    tn,fp,fn,tp=confusion_matrix(p.target,p.prediction,labels=[0,1]).ravel()
    assert (tn,fp,fn,tp)==(row.tn,row.fp,row.fn,row.tp)
    expected={"accuracy":accuracy_score(p.target,p.prediction),"precision":precision_score(p.target,p.prediction),"recall":recall_score(p.target,p.prediction),"f1":f1_score(p.target,p.prediction),"roc_auc":roc_auc_score(p.target,p.score),"average_precision":average_precision_score(p.target,p.score),"specificity":tn/(tn+fp)}
    for k,v in expected.items():assert np.isclose(v,getattr(row,k),atol=1e-12)
assert len(list((R/"results/plots").glob("*.png")))==9
for p in (R/"results/plots").glob("*.png"):assert p.read_bytes().startswith(b"\x89PNG")
# Verify imputation learns exclusively from a single training fold and does not change on holdout transform.
import runpy
module=runpy.run_path(str(R/"scripts/run_analysis.py"))
from sklearn.model_selection import StratifiedKFold
X=raw[module["NUM"]+module["CAT"]].copy()
for c in module["CAT"]:X[c]=X[c].map(lambda v:np.nan if pd.isna(v) else str(v))
tr=raw.index[raw.id.isin(train)]
fold_train,_=next(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X.loc[tr],raw.loc[tr,"num"].gt(0)))
fold=X.loc[tr].iloc[fold_train]
prep=module["prepare"](fold).fit(fold)
stats=prep.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
assert np.allclose(stats,fold[module["NUM"]].median().to_numpy())
prep.transform(X.loc[raw.id.isin(test)])
assert np.array_equal(stats,prep.named_transformers_["numeric"].named_steps["impute"].statistics_)
print("Verified source checksum, split separation, all six models' holdout metrics, fold-only imputation, and nine PNG charts.")
