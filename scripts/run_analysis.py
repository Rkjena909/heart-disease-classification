"""Reproduce the heart-disease classification benchmark."""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[1]
NUM=["age","trestbps","chol","thalch","oldpeak","ca"]
CAT=["sex","cp","fbs","restecg","exang","slope","thal"]
SEED=42

def prepare(frame, mean=False, minmax=False, drop_heavy=False):
    numeric=[c for c in NUM if c in frame.columns]
    categorical=[c for c in CAT if c in frame.columns]
    if drop_heavy:
        numeric=[c for c in numeric if c!="ca"]
        categorical=[c for c in categorical if c not in ["thal","slope"]]
    return ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="mean" if mean else "median",keep_empty_features=True)),("scale",MinMaxScaler() if minmax else StandardScaler())]),numeric),
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical)
    ],remainder="drop",verbose_feature_names_out=False)

def pipeline(frame, model, **kwargs):return Pipeline([("prepare",prepare(frame,**kwargs)),("model",model)])

def metrics(y, score):
    pred=(np.asarray(score)>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred),specificity=tn/(tn+fp),f1=f1_score(y,pred),roc_auc=roc_auc_score(y,score),average_precision=average_precision_score(y,score),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    out=ROOT/"results"
    for d in ["tables","plots","predictions"]:(out/d).mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/"data/heart_disease.csv")
    if not set(NUM+CAT+["num","id","dataset"]).issubset(raw.columns):raise ValueError("Required fields missing")
    if raw.num.isna().any() or not set(raw.num.unique()).issubset({0,1,2,3,4}):raise ValueError("Invalid target")
    if raw.id.duplicated().any():raise ValueError("Record IDs are not unique")
    X=raw[NUM+CAT].copy()
    for col in CAT:X[col]=X[col].map(lambda v:np.nan if pd.isna(v) else str(v))
    y=raw.num.gt(0).astype(int)
    train_idx,test_idx=train_test_split(np.arange(len(raw)),test_size=.2,random_state=SEED,stratify=y)
    Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx];ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(Xtr,ytr))
    scoring={"roc_auc":"roc_auc","average_precision":"average_precision","accuracy":"accuracy","f1":"f1","recall":"recall"}
    estimators={"LogisticRegression":LogisticRegression(max_iter=2000),"DecisionTree":DecisionTreeClassifier(random_state=SEED),"RandomForest":RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=1),"SVM":SVC(probability=True,random_state=SEED),"NeuralNetworkMLP":MLPClassifier(hidden_layer_sizes=(10,5),activation="relu",solver="adam",max_iter=500,early_stopping=True,random_state=SEED)}
    fitted={};cv_rows=[];held=[];predictions=[];warn=[]
    def record(name,pipe,cv_result):
        row={"model":name}
        for metric in scoring:
            row["cv_"+metric]=float(np.mean(cv_result["test_"+metric]));row["cv_"+metric+"_std"]=float(np.std(cv_result["test_"+metric],ddof=1))
        cv_rows.append(row)
        pipe.fit(Xtr,ytr);fitted[name]=pipe
        score=pipe.predict_proba(Xte)[:,1]
        held.append({"model":name,**metrics(yte,score)})
        for idx,label,s in zip(test_idx,yte,score):predictions.append({"model":name,"id":int(raw.iloc[idx].id),"site":raw.iloc[idx].dataset,"target":int(label),"score":float(s),"prediction":int(s>=.5)})
        print(name,row["cv_roc_auc"],held[-1]["roc_auc"],flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name,est in estimators.items():
            pipe=pipeline(Xtr,est)
            cv_result=cross_validate(pipe,Xtr,ytr,cv=splits,scoring=scoring,n_jobs=1,error_score="raise")
            record(name,pipe,cv_result)
        search=RandomizedSearchCV(pipeline(Xtr,estimators["RandomForest"]),param_distributions={"model__n_estimators":[100,200,300],"model__max_depth":[None,5,10,20],"model__min_samples_split":[2,5,10],"model__min_samples_leaf":[1,2,4],"model__max_features":["sqrt","log2"]},n_iter=20,cv=splits,scoring="roc_auc",random_state=SEED,n_jobs=1,error_score="raise",return_train_score=False)
        search.fit(Xtr,ytr)
        pd.DataFrame(search.cv_results_).to_csv(out/"tables/random_forest_search.csv",index=False)
        tuned=clone(search.best_estimator_)
        record("TunedRandomForest",tuned,cross_validate(tuned,Xtr,ytr,cv=splits,scoring=scoring,n_jobs=1,error_score="raise"))
        cv=pd.DataFrame(cv_rows).sort_values("cv_roc_auc",ascending=False)
        selected=str(cv.iloc[0].model)
        # Ablations use only the training records and the same folds. They do not replace the main preprocessing.
        prep=[]
        variants=[("median_standard",{},False),("mean_standard",{"mean":True},False),("median_minmax",{"minmax":True},False),("omit_heavily_missing",{"drop_heavy":True},False),("zero_bp_chol_as_missing",{},True)]
        for name,kwargs,zero_missing in variants:
            frame=Xtr.copy()
            if zero_missing:
                for c in ["chol","trestbps"]:frame[c]=frame[c].replace(0,np.nan)
            res=cross_validate(pipeline(frame,LogisticRegression(max_iter=2000),**kwargs),frame,ytr,cv=splits,scoring=scoring,n_jobs=1,error_score="raise")
            prep.append({"variant":name,**{f"cv_{m}":float(res["test_"+m].mean()) for m in scoring}})
        pd.DataFrame(prep).to_csv(out/"tables/preprocessing_comparison.csv",index=False)
        # Exploratory site evaluation: selected parameters are fixed. Selection used pooled training sites, so this is not nested external validation.
        site_rows=[]
        for site in sorted(raw.dataset.unique()):
            mask=raw.dataset.eq(site);pipe=clone(fitted[selected]);pipe.fit(X.loc[~mask],y.loc[~mask]);score=pipe.predict_proba(X.loc[mask])[:,1]
            site_rows.append({"model":selected,"site":site,"train_records":int((~mask).sum()),"evaluation_records":int(mask.sum()),**metrics(y.loc[mask],score)})
        pd.DataFrame(site_rows).to_csv(out/"tables/site_transfer_exploratory.csv",index=False)
        perm=permutation_importance(fitted[selected],Xte,yte,scoring="roc_auc",n_repeats=10,random_state=SEED,n_jobs=1)
        importance=pd.DataFrame({"feature":X.columns,"mean_auc_decrease":perm.importances_mean,"std":perm.importances_std}).sort_values("mean_auc_decrease",ascending=False)
        importance.to_csv(out/"tables/permutation_importance.csv",index=False)
        for w in caught:
            s=str(w.message)
            if s not in warn:warn.append(s)
    cv.to_csv(out/"tables/cross_validation_metrics.csv",index=False)
    hold=pd.DataFrame(held);hold.to_csv(out/"tables/holdout_metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"predictions/holdout_predictions.csv",index=False)
    pd.DataFrame({"id":raw.id,"split":np.where(np.isin(np.arange(len(raw)),train_idx),"train","holdout")}).to_csv(out/"tables/split_assignments.csv",index=False)
    missing=raw.isna().sum().rename("missing_records").rename_axis("field").reset_index();missing["missing_percent"]=missing.missing_records/len(raw)*100;missing.to_csv(out/"tables/missing_values.csv",index=False)
    summary={"source_sha256":hashlib.sha256((ROOT/"data/heart_disease.csv").read_bytes()).hexdigest(),"records":len(raw),"columns":len(raw.columns),"target_counts":y.value_counts().sort_index().to_dict(),"train_records":len(train_idx),"holdout_records":len(test_idx),"selected_model":selected,"selection_metric":"training five-fold ROC-AUC","random_seed":SEED,"rf_search_candidates":20,"rf_best_params":search.best_params_,"predictors":list(X.columns),"excluded_predictors":["id","dataset","num"],"warnings":warn,"execution_notes":["Tuning and model selection share CV folds; CV is not nested and tuned CV scores may be optimistic.","Holdout metrics are from one fixed pooled split at threshold 0.5.","Site-transfer experiment is exploratory; configuration selection already used pooled training-site records.","NeuralNetworkMLP uses scikit-learn, not the original TensorFlow implementation."]}
    (out/"experiment_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    historical=pd.DataFrame([{"model":n,"accuracy":a,"f1":f,"recall":rec,"roc_auc":auc} for n,a,f,rec,auc in [("LogisticRegression",.8152,.8396,.8165,.8379),("DecisionTree",.7772,.7960,.7339,.7870),("RandomForest",.8533,.8720,.8440,.9109),("SVM",.8261,.8505,.8349,.8232),("NeuralNetworkTensorFlow",.8098,.8372,.8257,.8334),("TunedRandomForest",.8587,.8796,.8716,.9118)]])
    historical.to_csv(out/"tables/historical_metrics.csv",index=False)
    sns.set_theme(style="whitegrid",palette="colorblind")
    def save(name):plt.tight_layout();plt.savefig(out/"plots"/name,dpi=150,bbox_inches="tight");plt.close()
    plt.figure(figsize=(7,4));plt.bar(["Negative","Positive"],[int((y==0).sum()),int((y==1).sum())]);plt.ylabel("Records");plt.title("Binary target distribution");save("class_distribution.png")
    m=missing.loc[missing.missing_records.gt(0)].sort_values("missing_percent")
    plt.figure(figsize=(8,5));plt.barh(m.field,m.missing_percent);plt.xlabel("Missing (%)");plt.title("Missingness in the supplied dataset");save("missing_values.png")
    plt.figure(figsize=(10,5));plt.barh(cv.model,cv.cv_roc_auc,xerr=cv.cv_roc_auc_std,color="#2878a0");plt.xlim(.5,1);plt.xlabel("Mean five-fold ROC-AUC (error bars: fold SD)");plt.title("Training cross-validation model comparison");save("model_comparison.png")
    fig,ax=plt.subplots(figsize=(8,6))
    for name,pipe in fitted.items():
        score=pipe.predict_proba(Xte)[:,1];fpr,tpr,_=roc_curve(yte,score);ax.plot(fpr,tpr,label=f"{name}: {roc_auc_score(yte,score):.3f}")
    ax.plot([0,1],[0,1],"k--");ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Pooled holdout ROC curves");ax.legend(fontsize=8);save("roc_curves.png")
    plt.figure(figsize=(8,6))
    for name,pipe in fitted.items():
        score=pipe.predict_proba(Xte)[:,1];p,rec,_=precision_recall_curve(yte,score);plt.plot(rec,p,label=f"{name}: AP {average_precision_score(yte,score):.3f}")
    plt.axhline(yte.mean(),ls="--",color="black",label="Positive share");plt.xlabel("Recall");plt.ylabel("Precision");plt.title("Pooled holdout precision–recall curves");plt.legend(fontsize=8);save("precision_recall_curves.png")
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(name,pipe) in zip(axes.flat,fitted.items()):
        cm=confusion_matrix(yte,pipe.predict_proba(Xte)[:,1]>=.5,labels=[0,1]);sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax);ax.set(title=name,xlabel="Predicted (0 / 1)",ylabel="Actual (0 / 1)")
    save("confusion_matrices.png")
    plt.figure(figsize=(10,6));plt.barh(importance.feature,importance.mean_auc_decrease,xerr=importance['std']);plt.xlabel("Holdout ROC-AUC decrease after permutation");plt.title(f"{selected}: predictor importance (10 repeats)");save("feature_importance.png")
    prep_df=pd.DataFrame(prep).sort_values("cv_roc_auc")
    plt.figure(figsize=(10,5));plt.barh(prep_df.variant,prep_df.cv_roc_auc);plt.xlim(.5,1);plt.xlabel("Training five-fold ROC-AUC");plt.title("Preprocessing sensitivity: Logistic Regression only");save("preprocessing_comparison.png")
    sites=pd.DataFrame(site_rows)
    plt.figure(figsize=(9,5));plt.bar(sites.site,sites.roc_auc);plt.ylim(0,1);plt.ylabel("ROC-AUC");plt.title("Exploratory transfer to a withheld collection site\nConfiguration selection used pooled sites");save("site_transfer.png")
    print(json.dumps(summary,indent=2),flush=True)
    return {"cv":cv,"holdout":hold,"preprocessing":prep_df,"sites":sites,"importance":importance,"summary":summary}

if __name__=="__main__":main()
