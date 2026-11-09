"""Part B: customer RFM segments, leakage-safe lapse modelling and retention shortlist."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
SNAPSHOT=pd.Timestamp('2025-10-01')
CUTOFF=pd.Timestamp('2025-09-01')

# Calculate precision, recall and F1 for the lapsed customer class (1).
def metrics(y,pred):
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',pos_label=1,zero_division=0)
    return {'precision':round(float(precision),4),'recall':round(float(recall),4),'f1':round(float(f1),4),'accuracy':round(float(accuracy_score(y,pred)),4),'confusion_matrix':confusion_matrix(y,pred).tolist()}

# Main workflow: customer segmentation first, then lapse prediction.
def main():
    customers=pd.read_csv('data/clean/customers.csv',parse_dates=['signup_date'])
    orders=pd.read_csv('data/clean/orders.csv',parse_dates=['order_ts'])
    products=pd.read_csv('data/clean/products.csv')
    orders=orders.merge(products[['product_id','category']],on='product_id',how='left',validate='many_to_one')
    completed=orders[orders.status=='completed'].copy()
    rfm=completed.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'))
    rfm['recency']=(SNAPSHOT-rfm.last_order).dt.days
    rfm=rfm.reset_index()[['customer_id','recency','frequency','monetary']]
    cols=['recency','frequency','monetary']
    scaled=StandardScaler().fit_transform(rfm[cols])
    diagnostics=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(scaled)
        diagnostics.append({'k':k,'inertia':round(float(km.inertia_),2),'silhouette':round(float(silhouette_score(scaled,km.labels_)),4)})
    pd.DataFrame(diagnostics).to_csv(OUT/'kmeans_scores.csv',index=False)
    chosen_k=4
    km=KMeans(n_clusters=chosen_k,n_init=10,random_state=42).fit(scaled)
    rfm['segment_id']=km.labels_
    profile=rfm.groupby('segment_id').agg(size=('customer_id','count'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean'))
    premium_id=int(profile.mean_monetary.idxmax())
    names={}
    for seg,row in profile.iterrows():
        if seg==premium_id: names[seg]='Premium'
        elif row.mean_frequency>12: names[seg]='Frequent Loyal'
        elif row.mean_recency>45: names[seg]='At Risk'
        else: names[seg]='Occasional'
    rfm['segment_name']=rfm.segment_id.map(names)
    rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm.to_csv(OUT/'customer_segments.csv',index=False)
    profile.assign(segment_name=profile.index.map(names)).to_csv(OUT/'segment_profile.csv')
    pca=PCA(n_components=2,random_state=42).fit(scaled)

    # IMPORTANT: only pre-September orders are allowed as model features.
    # September orders are used to build the answer (the lapse label), not inputs.
    # Build all features only using pre-September events; the September label is held out.
    before=orders[orders.order_ts<CUTOFF].copy()
    before_done=before[before.status=='completed'].copy()
    cohort=before_done.customer_id.drop_duplicates().sort_values()
    feature=before_done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
    feature['recency']=(CUTOFF-feature.last_order).dt.days
    feature=feature.drop(columns='last_order')
    cancellation=before.assign(cancelled=before.status.eq('cancelled')).groupby('customer_id').cancelled.mean()
    feature['cancellation_rate']=cancellation
    favorite=(before_done.groupby(['customer_id','category']).size().rename('count').reset_index().sort_values(['customer_id','count','category'],ascending=[True,False,True]).drop_duplicates('customer_id').set_index('customer_id').category)
    feature['favourite_category']=favorite
    feature=feature.join(customers.set_index('customer_id')[['city','age','gender','signup_date']],how='left')
    feature['tenure_days']=(CUTOFF-feature.signup_date).dt.days
    feature=feature.drop(columns='signup_date')
    september_customers=set(completed.loc[(completed.order_ts>=CUTOFF)&(completed.order_ts<SNAPSHOT),'customer_id'])
    feature['lapsed']=(~feature.index.isin(september_customers)).astype(int)
    feature.index.name='customer_id'
    feature.reset_index().to_csv(OUT/'lapse_cohort.csv',index=False)
    feature_columns=[x for x in feature.columns if x!='lapsed']
    X=feature[feature_columns]; y=feature.lapsed
    assert not X.isna().any().any(), X.isna().sum()
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    numeric=['frequency','monetary','avg_discount','avg_delivery','recency','cancellation_rate','age','tenure_days']
    categorical=['favourite_category','city','gender']
    def preprocessing():
        return ColumnTransformer([('numeric',StandardScaler(),numeric),('categorical',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),categorical)])
    log=GridSearchCV(Pipeline([('preprocess',preprocessing()),('model',LogisticRegression(max_iter=1000,random_state=42))]),{'model__C':[.01,.1,1,10,100]},scoring='f1',cv=5,n_jobs=-1)
    log.fit(Xtrain,ytrain)
    forest=GridSearchCV(Pipeline([('preprocess',preprocessing()),('model',RandomForestClassifier(n_estimators=100,random_state=42))]),{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},scoring='f1',cv=5,n_jobs=-1)
    forest.fit(Xtrain,ytrain)
    baseline=np.zeros(len(ytest),dtype=int)
    log_metrics=metrics(ytest,log.predict(Xtest)); forest_metrics=metrics(ytest,forest.predict(Xtest))
    # Select based on F1 on untouched test; compare tradeoffs in README.
    selected=forest.best_estimator_ if forest_metrics['f1']>=log_metrics['f1'] else log.best_estimator_
    label='Random Forest' if selected is forest.best_estimator_ else 'Logistic Regression'
    assert max(log_metrics['f1'],forest_metrics['f1'])>=.55
    joblib.dump(selected,OUT/'lapse_model.joblib')
    reloaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(reloaded.predict(Xtest),selected.predict(Xtest))
    # Training-only oversampling; preserve raw test cohort completely.
    train_frame=Xtrain.copy(); train_frame['label']=ytrain
    non_lapse=train_frame[train_frame.label==0]
    lapse=train_frame[train_frame.label==1]
    extra=lapse.sample(n=len(non_lapse)-len(lapse),replace=True,random_state=42)
    balanced=pd.concat([train_frame,extra]).sample(frac=1,random_state=42)
    balanced_log=Pipeline([('preprocess',preprocessing()),('model',LogisticRegression(C=log.best_params_['model__C'],max_iter=1000,random_state=42))])
    balanced_log.fit(balanced[feature_columns],balanced.label)
    balanced_metrics=metrics(ytest,balanced_log.predict(Xtest))
    segment=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
    test_results=pd.DataFrame({'customer_id':Xtest.index,'lapse_probability':selected.predict_proba(Xtest)[:,1], 'actually_lapsed':ytest.to_numpy()}).set_index('customer_id')
    premium_test=test_results.join(segment[['is_premium']],how='left')
    retention=premium_test[premium_test.is_premium.fillna(False)].sort_values('lapse_probability',ascending=False).head(10).reset_index()[['customer_id','lapse_probability','actually_lapsed']]
    retention.to_csv(OUT/'retention_list.csv',index=False)
    fn=forest.best_estimator_.named_steps['preprocess'].get_feature_names_out()
    imp=forest.best_estimator_.named_steps['model'].feature_importances_
    importance=pd.DataFrame({'feature':fn,'importance':imp}).sort_values('importance',ascending=False)
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    log_coeff=pd.DataFrame({'feature':log.best_estimator_.named_steps['preprocess'].get_feature_names_out(),'coefficient':log.best_estimator_.named_steps['model'].coef_[0]})
    log_coeff.to_csv(OUT/'logistic_coefficients.csv',index=False)
    report={
      'rfm_count':len(rfm),'rfm_median_recency':float(rfm.recency.median()),'rfm_mean_frequency':float(rfm.frequency.mean()),'rfm_mean_monetary':float(rfm.monetary.mean()),
      'k_selected':chosen_k,'premium_segment_id':premium_id,'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'cohort':len(y),'lapsed':int(y.sum()),'train':len(ytrain),'test':len(ytest),
      'training_scaler_frequency_mean':float(log.best_estimator_.named_steps['preprocess'].named_transformers_['numeric'].mean_[numeric.index('frequency')]), 'cohort_frequency_mean':float(X.frequency.mean()),
      'baseline':metrics(ytest,baseline),'logistic_best_params':log.best_params_,'logistic_cv_f1':float(log.best_score_),'logistic_test':log_metrics,
      'forest_best_params':forest.best_params_,'forest_cv_f1':float(forest.best_score_),'forest_test':forest_metrics,
      'balanced_logistic_test':balanced_metrics,'selected_model':label,'retention_true_lapsed':int(retention.actually_lapsed.sum()),'retention_count':len(retention),
      'top_importance':importance.head(5).to_dict(orient='records'),'logistic_coefficients':log_coeff.sort_values('coefficient',key=abs,ascending=False).head(8).to_dict(orient='records')
    }
    (OUT/'part_b_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
