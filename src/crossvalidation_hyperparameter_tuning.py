import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

from load_data import load_data
from preprocess import preprocess_data
from plot import plot_precision_recall_curve


def get_models_and_params():
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [5, 10, None], "min_samples_split": [2, 5, 10]}
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "saga"]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.3]}
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.3]}
        )
    }

def stratified_cv_tuning(X_train, y_train, models_params):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for name, (model, params) in models_params.items():
        print(f"Starting CV tuning: {name}")
        start_time = time.time()
        grid = GridSearchCV(model, params, cv=skf, scoring='roc_auc_ovr', n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        end_time = time.time()
        avg_time = (end_time - start_time) / skf.get_n_splits()
        print(f"Best params: {grid.best_params_}")
        print(f"Best mean CV AUC ROC: {grid.best_score_:.4f}")
        results.append({
            "model": name,
            "best_estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_cv_auc": grid.best_score_,
            "avg_cv_time_per_fold": avg_time
        })
    return results

def select_best_model(results):
    return max(results, key=lambda x: x['best_cv_auc'])

def evaluate_on_test(estimator, X_test, y_test):
    y_prob = estimator.predict_proba(X_test)
    y_pred = estimator.predict(X_test)

    # Determine if the problem is binary or multiclass
    if len(np.unique(y_test)) == 2:
        # Binary classification
        auc = roc_auc_score(y_test, y_prob[:, 1])  # Use probabilities of the positive class
        avg_method = 'binary'
    else:
        # Multiclass classification
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        avg_method = 'macro'

    f1 = f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    rec = recall_score(y_test, y_pred, average=avg_method)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return dict(auc_roc=auc, f1=f1, precision=prec, recall=rec, accuracy=acc, confusion_matrix=cm)

def save_results_to_dir(summary, cm, labels, cv_results, time_prefix, output_dir="output", runtime=None):

    results_df = pd.DataFrame([
        {"model": r["model"], "best_params": r["best_params"], "best_cv_auc": r["best_cv_auc"], "avg_cv_time_per_fold": r["avg_cv_time_per_fold"]}
        for r in cv_results
    ])
    cv_file = os.path.join(output_dir, f"{time_prefix}_cv_results.csv")
    results_df.to_csv(cv_file, index=False)
    print(f"CV tuning summary saved: {cv_file}")

    summary_copy = summary.copy()
    if runtime is not None:
        summary_copy["runtime_seconds"] = runtime
    summary_df = pd.DataFrame([summary_copy])
    summary_file = os.path.join(output_dir, f"{time_prefix}_test_metrics.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Test metrics saved to {summary_file}")

    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_file = os.path.join(output_dir, f"{time_prefix}_confusion_matrix.csv")
    cm_df.to_csv(cm_file)
    print(f"Confusion matrix saved to {cm_file}")


def export_global_feature_importance(X_train, clf, prefix='', output_dir='.'):
    # Get feature importances
    importances = clf.feature_importances_

    feature_importances_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })

    # Sort by importance descending
    feature_importances_df_sorted = feature_importances_df.sort_values(by='importance', ascending=False)

    # Print sorted features and their importance
    for _, row in feature_importances_df_sorted.iterrows():
        print(f"{row['feature']}: {row['importance']}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export feature importance to text file
    output_path = os.path.join(output_dir, f"{prefix}_feature_importance.txt")
    feature_importances_df_sorted.to_csv(output_path, index=False, sep='\t')


# ---------- MAIN PIPELINE (to run) ----------
def run_pipeline(X, y, output_dir="output"):
    
    # Split data in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    make_binary = True  # Set to True for binary classification
    X_train, X_test, y_train, y_test, target_label_encoder = preprocess_data(X_train, X_test, y_train, y_test, make_binary=make_binary)


    # Perform Cross-Validation with Hyperparameter Tuning
    models_params = get_models_and_params()
    cv_results = stratified_cv_tuning(X_train, y_train, models_params)


    # Choose best model by CV ROC AUC
    best = select_best_model(cv_results)
    print(f"\nBest model: {best['model']}")
    print("Best parameters:", best["best_params"])

    # Retrain on full train set
    best_estimator = best["best_estimator"]
    best_estimator.fit(X_train, y_train)

    # Evaluate on test set
    test_metrics = evaluate_on_test(best_estimator, X_test, y_test)

    # Save results and confusion matrix
    summary_metrics = {
        "model": best['model'],
        "best_params": best['best_params'],
        "auc_roc": test_metrics['auc_roc'],
        "f1": test_metrics['f1'],
        "precision": test_metrics['precision'],
        "recall": test_metrics['recall'],
        "accuracy": test_metrics['accuracy']
    }
    labels = np.unique(y_test)
    
    # Save results to dir
    os.makedirs(output_dir, exist_ok=True)
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_to_dir(summary_metrics, test_metrics['confusion_matrix'], labels, cv_results, time_prefix, output_dir)

    # Plot Precision-Recall curve
    plot_precision_recall_curve(best_estimator, X_test, y_test, output_dir, prefix=time_prefix)

    # Plot learning curve
    # plot_learning_curve(best_estimator, X_train, y_train, output_dir, prefix=time_prefix, cv=3)

    #Export feature importance
    export_global_feature_importance(X_train, best_estimator, prefix=time_prefix, output_dir=output_dir)



if __name__ == "__main__":
    X_train, X_test, y_train = load_data()
    run_pipeline(X_train, y_train.iloc[:,-1], output_dir="output")
