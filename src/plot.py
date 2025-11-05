import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


def plot_precision_recall_curve(estimator, X_test, y_test, output_dir="output", prefix=""):
    y_prob = estimator.predict_proba(X_test)
    plt.figure()
    if len(set(y_test)) > 2:  # multiclass case (one-vs-rest for each class)
        for i, label in enumerate(estimator.classes_):
            precision, recall, _ = precision_recall_curve((y_test == label).astype(int), y_prob[:, i])
            avg_prec = average_precision_score((y_test == label).astype(int), y_prob[:, i])
            plt.plot(recall, precision, label=f"Class {label} (AP={avg_prec:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Multiclass Precision-Recall Curve (One-vs-Rest)")
        plt.legend()
    else:  # binary case
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:, 1])
        avg_prec = average_precision_score(y_test, y_prob[:, 1])
        plt.plot(recall, precision, label=f"AP={avg_prec:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

        thresholds = np.append(thresholds, np.nan)
        df = pd.DataFrame({
            'threshold': thresholds,
            'precision': precision,
            'recall': recall
        })
        out_file = f"{prefix}_precision_recall_curve.csv"
        df.to_csv(os.path.join(output_dir, out_file), index=False)

    plt.tight_layout()
    pr_file = os.path.join(output_dir, f"{prefix}_precision_recall_curve.png")
    plt.savefig(pr_file)
    plt.close()
    print(f"Precision-Recall curve saved to {pr_file}")


def plot_learning_curve(estimator, X, y, output_dir="output", prefix="", cv=3, scoring="roc_auc"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel(scoring.capitalize())
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    lc_file = f"{prefix}_learning_curve.png"
    plt.savefig(os.path.join(output_dir, lc_file))
    plt.close()
    print(f"Learning curve plot saved to {os.path.join(output_dir, lc_file)}")


def plot_feature_distribution(X_train, y_train, feature_name, output_dir='output'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataframe
    df = pd.DataFrame(X_train.copy())
    df['target'] = y_train
    
    plt.figure(figsize=(8,5))

    if pd.api.types.is_numeric_dtype(df[feature_name]):
        # Remove outliers based on quantiles
        q_low = df[feature_name].quantile(0.05)
        q_high = df[feature_name].quantile(0.95)
        df = df[(df[feature_name] >= q_low) & (df[feature_name] <= q_high)]
        sns.histplot(data=df, x=feature_name, hue='target', element='step', stat='density', common_norm=False)
    else:
        # For categoricals with >10 values: show top 10, rest as 'Other'
        top_categories = df[feature_name].value_counts().nlargest(10).index
        df = df[df[feature_name].isin(top_categories)]  # Discard all other categories
        ax= sns.countplot(data=df, x=feature_name, hue='target')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.title(f'Distribution of {feature_name} by target')
    plt.tight_layout()
    # Add timestamp prefix to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(output_dir, f'{timestamp}_{feature_name}_by_target.png')
   
    plt.savefig(save_path)
    plt.close()
    print(f'Saved plot to {save_path}')

def run_feature_distribution_plots():
    from load_and_clean_data import load_data

    X_train, X_test, y_train = load_data()
    y_train = y_train.iloc[:, -1]

    y_train = y_train.replace({
                'functional needs repair': 'Needs repair or non functional',
                'non functional': 'Needs repair or non functional'
            })

    for feature_name in X_train.columns:
        plot_feature_distribution(X_train, y_train, feature_name)

if __name__ == "__main__":
    run_feature_distribution_plots()