import os
import warnings
import mne
import numpy as np
import antropy
import matplotlib.pyplot as plt
import seaborn as sns
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------
# Global Configurations
# ---------------------------------------------------------------------

# Remove Verbose Logs
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Dataset Path
DATA_DIR = "/Users/gleblevashov/Visual Studio Projects/BME772 Project/files"

# Select EEG Channels
CHANNELS_LOAD = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]

# Select Key Motor Channels for Feature Extraction
CHANNELS_FEATS = ["FC3", "C3", "CP3", "FC4", "C4", "CP4", "Cz"]

# Select Task 2 (Imagine Opening and Closing Left or Right Fist) Motor Imagery Runs
RUNS_GROUP = (4, 8, 12)

# Set Subject Range
SUBJECT_IDS = range(1, 110)

# Set Entropy Methods
METHODS = ["sampen", "permen", "appen"]

# Minimum Number of Trials per Class
MIN_TRIALS_PER_CLASS = 15

# Set a Single Wide Motor Band (Mu+Beta)
BANDS = {
    "motor": (8.0, 30.0),
}

# Output Directory for Visualizations
OUTPUT_DIR = "/Users/gleblevashov/Visual Studio Projects/BME772 Project/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------

def load_subject_raw(subject, runs):
    """
    Load raw data for one subject in EDF format (the dataset's raw datatype) and apply:
      - EEGBCI channel standardization
      - 1–40 Hz band-pass
      - Cross-Spectral Density (CSD)
    """
    
    subject_str = f"S{subject:03d}"
    run0 = runs[0]

    # Get path's nested and flat layouts to check which format is loaded
    path_nested = os.path.join(DATA_DIR, subject_str, f"{subject_str}R{run0:02d}.edf")
    path_flat   = os.path.join(DATA_DIR, f"{subject_str}R{run0:02d}.edf")

    if os.path.exists(path_nested):
        path_fmt = os.path.join(DATA_DIR, subject_str, f"{subject_str}R{{run:02d}}.edf")
    elif os.path.exists(path_flat):
        path_fmt = os.path.join(DATA_DIR, f"{subject_str}R{{run:02d}}.edf")
    else:
        return None

    try:
        raws = []
        for run in runs:
            fname = path_fmt.format(run=run)
            raw = mne.io.read_raw_edf(fname, preload=True, verbose="ERROR")
            raws.append(raw)

        raw = mne.concatenate_raws(raws)

        # Standardize EEGBCI Channel Names
        eegbci.standardize(raw)

        # Apply Standard Montage Necessary for CSD
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Load All Available EEG Channels for CSD
        raw.pick_channels(CHANNELS_LOAD)

        # Apply Temporal Filtering from 1-40 Hz
        raw.filter(1.0, 40.0, fir_design="firwin", verbose="ERROR")

        # Apply Spatial Filtering Through Current Source Density (CSD)
        raw = mne.preprocessing.compute_current_source_density(raw)

        # Choose Only Key Motor Channels for Future Feature Extraction
        raw.pick_channels(CHANNELS_FEATS)

        return raw

    except Exception:
        return None

def epoch_left_right_MI(raw, tmin=0.5, tmax=3.5):
    """
    Creates 3 Second Epochs for T1 (Left Fist Imagery) and T2 (Right Fist Imagery).
    Rejection Skipped to Avoid Losing Trials.
    """
    events, event_id_all = mne.events_from_annotations(raw, verbose="ERROR")

    if "T1" not in event_id_all or "T2" not in event_id_all:
        raise ValueError("Missing T1 or T2 events")

    event_id = {"T1": event_id_all["T1"], "T2": event_id_all["T2"]}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    return epochs

# ---------------------------------------------------------------------
# Entropy Feature Extraction & Pipeline
# ---------------------------------------------------------------------

class EntropyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract Entropy Features Using Scikit-learn Compatible Transformer.

    Inputted Data:  X with Shape (n_epochs, n_channels, n_times)
    Output Data: (n_epochs, n_channels * n_bands)
    """

    # Initialize the Extractor
    def __init__(self, sfreq, method="sampen", bands=BANDS):
        self.sfreq = sfreq
        self.method = method
        self.bands = bands

    def fit(self, X, y=None):
        return self

    # Apply Filters to the Data and Calculate Entropies
    def transform(self, X):
        n_epochs, n_channels, _ = X.shape
        features = []

        for _, (l_freq, h_freq) in self.bands.items():
            # Band-pass Filter Per Band
            X_band = mne.filter.filter_data(
                X,
                self.sfreq,
                l_freq,
                h_freq,
                verbose="ERROR",
                method="iir", # Choose IIR Filter Due to Faster Computational Speed Compared to FIR Filters
            )

            # Initialize Array to Store Specific Band Features
            band_feats = np.zeros((n_epochs, n_channels))

            # Loop Through Each Epoch in Each Channel to Calculate Entropy
            for i in range(n_epochs):
                for c in range(n_channels):
                    sig = X_band[i, c, :]

                    # Measure Sample Entropy of Order 2 with Standard Chebyshev Metric 
                    if self.method == "sampen":
                        val = antropy.sample_entropy(sig, order=2, metric="chebyshev")
                    
                    # Measure Normalized Permuatation Entropy of Order 5 with Delay of 1
                    elif self.method == "permen":
                        val = antropy.perm_entropy(
                            sig, order=5, delay=1, normalize=True
                        )
                    
                    # Measure Approximate Entropy of Order 2 with Standard Chebyshev Metric
                    elif self.method == "appen":
                        val = antropy.app_entropy(sig, order=2, metric="chebyshev")
                    else:
                        raise ValueError(f"Unknown method {self.method}")

                    band_feats[i, c] = val

            # Add Features to the band_feats Array
            features.append(band_feats)

        return np.hstack(features)


def get_pipeline(sfreq, method):
    """
    Build the Full Machine Learning pipeline: Entropy -> Scaling -> LDA(with shrinkage)
    """
    return Pipeline(
        steps=[
            ("entropy", EntropyFeatureExtractor(sfreq=sfreq, method=method)),
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def evaluate_subject(epochs, method):
    """
    5-fold Cross-Validation for Accuracy Measurement of Pipeline for One Subject & Feature Method
    """
    X = epochs.get_data()
    y_codes = epochs.events[:, 2]
    t1_id = epochs.event_id["T1"]
    t2_id = epochs.event_id["T2"]
    y = np.where(y_codes == t1_id, 0, 1)

    sfreq = epochs.info["sfreq"]
    clf = get_pipeline(sfreq, method)

    # Calculate Cross-Validation Scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)  # n_jobs=1 Due to Antropy
    return scores.mean(), X, y, clf


def calibration_analysis(X, y, clf, trials_per_class_list=(5, 10, 20, 40)):
    """
    Simulate short calibration:

      - Take n Trials per Class for Training
      - Test on Remainder
      - Return Accuracy as a Function of Trial Count

    Determine How Much Training Data is Required From a User Before the System Becomes Reliable
    """
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    # Select Random Trials
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    results = {}

    for n in trials_per_class_list:
        # Check if Subject Has Enough Recorded Trials
        if len(idx_0) < n or len(idx_1) < n:
            continue

        # Separate 'n' Trials For Training and Use the Rest for Testing
        train_idx = np.concatenate([idx_0[:n], idx_1[:n]])
        test_idx = np.concatenate([idx_0[n:], idx_1[n:]])

        if len(test_idx) == 0:
            continue

        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])

        results[n] = acc

    return results


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

def plot_group_results(method_scores, save_path=None):
    """
    Generate Bar Plot of Mean Accuracy Across Subjects for Each Entropy Method
    """
    methods = []
    means = []
    stds = []

    for m in METHODS:
        scores = method_scores.get(m, [])
        if len(scores) == 0:
            continue
        methods.append(m.upper())
        means.append(np.mean(scores)) # Average Accuracy Across All Subjects
        stds.append(np.std(scores)) # Variance Across Subjects

    if not means:
        print("[WARN] No valid subjects to plot group results.")
        return

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "group_accuracy.png")

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    bars = ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.8)
    
    # Draw Horizontal Line Representing Guessing Threshold for Binary Classification
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")

    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Group Accuracy Across Entropy Methods")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n[FIG] Saved group accuracy bar plot -> {save_path}")


def plot_calibration_curves(calib_results, subject_id, epoch_len=3.0, save_path=None):
    """
    Plot Calibration Curve of the System for Demo Subject

    Show Trade-Off Between Calibration Time and Pipeline Accuracy
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "calibration_curve_demo_subject.png")

    plt.figure(figsize=(8, 5))
    for method, res in calib_results.items():
        if not res:
            continue
        trials = sorted(res.keys())
        accs = [res[n] for n in trials]
        
        # (2 Classes * Trials Per Class * 3 Seconds Per Epoch) / 60 Seconds
        mins = [(2 * n * epoch_len) / 60.0 for n in trials]

        plt.plot(mins, accs, marker="o", linewidth=2, label=method.upper())

    plt.axhline(0.5, color="black", linestyle="--", label="Chance")
    plt.ylim(0.3, 1.05)
    plt.xlabel("Calibration Time (minutes)")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Curve – Subject S{subject_id:03d}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[FIG] Saved calibration curve -> {save_path}")


def plot_confusion_matrix_for_subject(
    epochs,
    method="permen",
    save_path=None,
):
    """
    Confusion Matrix for a Single Subject (S001) Through Permutation Entropy

    Shows if Classifier is Biased Towards One Hand or the Other
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "confusion_matrix_permen.png")

    X = epochs.get_data()
    y_codes = epochs.events[:, 2]
    y = np.where(y_codes == epochs.event_id["T1"], 0, 1)

    # Use 70/30 Split (70% of Data for Training, 30% for Testing) for Quick Diagnostic Visualization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    sfreq = epochs.info["sfreq"]
    clf = get_pipeline(sfreq, method)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute + Normalize Matrix for Rows' Sum to Equal 1
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Left Fist (T1)", "Right Fist (T2)"],
        yticklabels=["Left Fist (T1)", "Right Fist (T2)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix – {method.upper()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[FIG] Saved confusion matrix -> {save_path}")


def plot_spatial_weights_for_subject(
    epochs,
    method="permen",
    save_path=None,
):
    """
    Topological Scalp Map of LDA Spatial Weights for a Single Subject.

    Height Weights in Left (C3) and Right (C4) Areas of Motor Cortex Signify Greatest Activity In These Regions
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "topomap_spatial_weights_permen.png")

    X = epochs.get_data()
    y_codes = epochs.events[:, 2]
    y = np.where(y_codes == epochs.event_id["T1"], 0, 1)

    sfreq = epochs.info["sfreq"]
    clf = get_pipeline(sfreq, method)
    clf.fit(X, y)

    # Get Weight Coefficients from Trained LDA Model
    lda = clf.named_steps["clf"]
    weights = lda.coef_[0]  # shape: (n_features,)

    n_ch = len(epochs.info["ch_names"])
    if weights.shape[0] != n_ch:
        print(
            f"[WARN] #features ({weights.shape[0]}) != #channels ({n_ch}); "
            "cannot map 1:1 to topomap."
        )
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    mne.viz.plot_topomap(
        weights,
        epochs.info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        contours=0,
    )
    ax.set_title(f"LDA Spatial Weights ({method.upper()})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[FIG] Saved spatial weights topomap -> {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_analysis():
    print("=== Calibration Curve Calculations ===")

    method_scores = {m: [] for m in METHODS}

    # Select S001 as Demo Subject
    demo_subject_id = 1
    demo_epochs = None
    calib_results_demo = {m: {} for m in METHODS}

    for sub in SUBJECT_IDS:
        raw = load_subject_raw(sub, RUNS_GROUP)
        if raw is None:
            continue

        try:
            epochs = epoch_left_right_MI(raw)
            if len(epochs) < MIN_TRIALS_PER_CLASS * 2:
                continue
        except Exception:
            continue

        # Save Demo Subject's Epochs
        if sub == demo_subject_id and demo_epochs is None:
            demo_epochs = epochs

        # Calculate Each Entropy
        for method in METHODS:
            acc, X, y, clf = evaluate_subject(epochs, method)
            method_scores[method].append(acc)

            if sub == demo_subject_id and not calib_results_demo[method]:
                print(f"\nSubject S{demo_subject_id:03d} Calibration Curve ({method}):")
                cal_res = calibration_analysis(X, y, clf)
                for n, res_acc in cal_res.items():
                    mins = (2 * n * 3.0) / 60.0
                    print(f"  {n} trials/class ({mins:.1f} min): {res_acc:.3f}")
                calib_results_demo[method] = cal_res

    print("\n=== Final Group Results (Mean Accuracy) ===")
    for m in METHODS:
        if method_scores[m]:
            print(f"{m.upper()}: {np.mean(method_scores[m]):.3f} (N={len(method_scores[m])})")
        else:
            print(f"{m.upper()}: No valid subjects.")

    return method_scores, demo_subject_id, demo_epochs, calib_results_demo

if __name__ == "__main__":

    method_scores, demo_subject_id, demo_epochs, calib_results_demo = run_analysis()

    plot_group_results(method_scores)

    if demo_epochs is not None:
        plot_calibration_curves(calib_results_demo, demo_subject_id, epoch_len=3.0)
        plot_confusion_matrix_for_subject(demo_epochs, method="permen")
        plot_spatial_weights_for_subject(demo_epochs, method="permen")
    else:
        print("[WARN] No valid demo subject found for detailed visualizations.")
