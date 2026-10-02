"""Run the full experiment: preprocessing, SMOTE, Bayesian Network, comparison
models, probabilistic inference examples and independent validation.

Usage:
    python main.py                      # full run (all steps)
    python main.py --sample-frac 0.05   # quick run on 5% of the records
    python main.py --skip-baselines     # Bayesian Network only
    python main.py --skip-external      # do not run the CMC validation
"""

import argparse
import json

import pandas as pd

from src import config
from src.balancing import smote_balance
from src.baselines import baseline_models
from src.bayesian_network import build_network, fit_mle, predict, query
from src.data_preprocessing import class_distribution, load_raw_data, preprocess
from src.evaluation import METRIC_NAMES, Timer, classification_metrics, roc_from_labels
from src.external_validation import load_cmc
from src import plots

TIME_COLUMN = "Total Time (s)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="use a stratified fraction of the records (for quick tests)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="skip Logistic Regression, KNN and Random Forest")
    parser.add_argument("--skip-external", action="store_true",
                        help="skip the independent validation on data/cmc.csv")
    parser.add_argument("--keep-test-distribution", action="store_true",
                        help="evaluate on the test file without SMOTE (original class ratio)")
    return parser.parse_args()


def stratified_sample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    return (df.groupby(config.TARGET, group_keys=False)
              .sample(frac=frac, random_state=config.RANDOM_STATE)
              .reset_index(drop=True))


def main():
    args = parse_args()
    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)

    # 1. Data --------------------------------------------------------------
    print("Loading data ...")
    train_raw, test_raw = load_raw_data()
    class_distribution(train_raw).to_csv(out / "class_distribution.csv")
    print(class_distribution(train_raw), "\n")

    print("Preprocessing ...")
    train, test = preprocess(train_raw, test_raw)
    del train_raw, test_raw
    if args.sample_frac:
        train = stratified_sample(train, args.sample_frac)
        test = stratified_sample(test, args.sample_frac)
    print(f"  training records: {len(train):,}   test records: {len(test):,}\n")

    # 2. SMOTE -------------------------------------------------------------
    print("Balancing classes with SMOTE ...")
    train_bal = smote_balance(train)
    test_eval = test if args.keep_test_distribution else smote_balance(test)
    print("  training class counts:", train_bal[config.TARGET].value_counts().sort_index().to_dict())
    print("  test class counts:    ", test_eval[config.TARGET].value_counts().sort_index().to_dict(), "\n")

    X_train = train_bal.drop(columns=[config.TARGET])
    y_train = train_bal[config.TARGET]
    X_test = test_eval.drop(columns=[config.TARGET])
    y_test = test_eval[config.TARGET]

    # 3. Bayesian Network --------------------------------------------------
    print("Training the Bayesian Network (MLE) ...")
    rows = {}
    bn = build_network(config.EVAL_PARENTS)
    with Timer() as t_fit:
        fit_mle(bn, train_bal)
    with Timer() as t_pred:
        y_pred_bn = predict(bn, X_test)
    bn_scores = classification_metrics(y_test, y_pred_bn, average="weighted")
    rows["Proposed Bayesian Network"] = {**bn_scores, TIME_COLUMN: t_fit.seconds + t_pred.seconds}

    # 4. Comparison models -------------------------------------------------
    if not args.skip_baselines:
        for name, model in baseline_models().items():
            print(f"Training {name} ...")
            with Timer() as t_fit:
                model.fit(X_train, y_train)
            with Timer() as t_pred:
                y_pred = model.predict(X_test)
            scores = classification_metrics(y_test, y_pred, average="weighted")
            rows[name] = {**scores, TIME_COLUMN: t_fit.seconds + t_pred.seconds}

    table = pd.DataFrame(rows).T[METRIC_NAMES + [TIME_COLUMN]]
    table.to_csv(out / "model_performance_and_times.csv")
    print("\nModel comparison (test data):")
    print(table.round(4).to_string(), "\n")
    plots.plot_metric_comparison(table, METRIC_NAMES, out / "performance_metrics_chart.png")
    plots.plot_total_time(table, TIME_COLUMN, out / "total_time_chart.png")

    fpr, tpr, roc_auc = roc_from_labels(y_test, y_pred_bn)
    plots.plot_roc(fpr, tpr, roc_auc, out / "roc_curve.png")
    print(f"Bayesian Network AUC: {roc_auc:.4f}\n")

    # 5. Probabilistic inference examples ----------------------------------
    print("Probabilistic inference examples ...")
    bn_inf = fit_mle(build_network(config.INFERENCE_PARENTS), train_bal)
    example = query(bn_inf, config.EVIDENCE_EXAMPLE)
    plots.plot_inference_table(config.EVIDENCE_EXAMPLE, example, out / "inference_example.png")
    posteriors = [query(bn_inf, ev) for ev in config.EVIDENCE_SETS]
    plots.plot_evidence_sets(config.EVIDENCE_SETS, posteriors, out / "inference_evidence_sets.png")
    inference_rows = [{"evidence": json.dumps(ev), "P(corona_result=0)": p[0], "P(corona_result=1)": p[1]}
                      for ev, p in zip([config.EVIDENCE_EXAMPLE] + config.EVIDENCE_SETS, [example] + posteriors)]
    pd.DataFrame(inference_rows).to_csv(out / "inference_results.csv", index=False)
    for r in inference_rows:
        print(f"  {r['evidence']}: P(positive) = {r['P(corona_result=1)']:.4f}")
    print()

    # 6. Independent validation --------------------------------------------
    if args.skip_external:
        print("Independent validation skipped (--skip-external).")
    elif not config.CMC_FILE.exists():
        print(f"Independent validation skipped: {config.CMC_FILE.relative_to(config.ROOT)} not found "
              "(see data/README.md).")
    else:
        print("Independent validation on data/cmc.csv ...")
        cmc = load_cmc()
        bn_ext = fit_mle(build_network(config.EXTERNAL_PARENTS), train_bal)
        y_true_ext = cmc[config.TARGET]
        y_pred_ext = predict(bn_ext, cmc.drop(columns=[config.TARGET]))
        ext_scores = classification_metrics(y_true_ext, y_pred_ext, average="binary")
        comparison = pd.DataFrame({"Training Dataset": bn_scores, "Independent Dataset": ext_scores}).T
        comparison.to_csv(out / "training_vs_independent.csv")
        print(f"  records: {len(cmc):,}")
        print(comparison.round(4).to_string())
        plots.plot_training_vs_independent(bn_scores, ext_scores, out / "training_vs_independent.png")
        plots.plot_confusion_matrix(y_true_ext, y_pred_ext, "Confusion Matrix (Independent Dataset)",
                                    out / "confusion_matrix_independent.png")

    print(f"\nResults saved in {out}")


if __name__ == "__main__":
    main()
