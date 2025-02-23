import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
import argparse
import os

# Define file paths
CONFIG = {
    "X_train": "D:\GUVI\GUVI Final project\Kannada-MNIST/X_kannada_MNIST_train.npz",
    "Y_train": "D:\GUVI\GUVI Final project\Kannada-MNIST/y_kannada_MNIST_train.npz",
    "X_test": "D:\GUVI\GUVI Final project\Kannada-MNIST/X_kannada_MNIST_test.npz",
    "Y_test": "D:\GUVI\GUVI Final project\Kannada-MNIST/y_kannada_MNIST_test.npz",
}


def load_data():
    """Load the dataset from .npz files."""
    try:
        X_train = np.load(CONFIG["X_train"])['arr_0']
        Y_train = np.load(CONFIG["Y_train"])['arr_0']
        X_test = np.load(CONFIG["X_test"])['arr_0']
        Y_test = np.load(CONFIG["Y_test"])['arr_0']
        print("Data successfully loaded.")
        return X_train, Y_train, X_test, Y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)


def preprocess_data(X_train, X_test):
    """Flatten images and normalize pixel values."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_train_flat, X_test_flat


def apply_pca(X_train_flat, X_test_flat, n_components):
    """Apply PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train_flat), pca.transform(X_test_flat)


def train_and_evaluate(models, X_train_pca, X_test_pca, Y_train, Y_test, n_components):
    """Train models and evaluate their performance."""
    results = []

    # Binarize labels for ROC-AUC
    Y_test_bin = label_binarize(Y_test, classes=np.unique(Y_train))

    for name, model in models.items():
        print(f"Training {name} with {n_components} PCA components...")
        model.fit(X_train_pca, Y_train)
        Y_pred = model.predict(X_test_pca)

        # Compute metrics
        report = classification_report(Y_test, Y_pred, output_dict=True)
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        # Compute ROC-AUC only if applicable
        roc_auc = None
        if hasattr(model, "predict_proba"):
            Y_prob = model.predict_proba(X_test_pca)
            roc_auc = roc_auc_score(Y_test_bin, Y_prob, multi_class='ovr')

        results.append({
            "Model": name,
            "PCA Components": n_components,
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-score": report["weighted avg"]["f1-score"],
            "ROC-AUC": roc_auc
        })

    return results


def plot_results(df_results):
    """Plot model performance vs PCA components."""
    sns.set_style("whitegrid")
    metrics = ["Precision", "Recall", "F1-score", "ROC-AUC"]

    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.lineplot(data=df_results, x="PCA Components", y=metric, hue="Model", marker="o")
        plt.title(f"{metric} vs PCA Components")
        plt.xlabel("Number of PCA Components")
        plt.ylabel(metric)
        plt.legend(title="Model")
    plt.tight_layout()
    plt.show()


def main():
    """Main function to execute the script."""
    # Load Data
    X_train, Y_train, X_test, Y_test = load_data()

    # Preprocess Data
    X_train_flat, X_test_flat = preprocess_data(X_train, X_test)

    # Define classifiers
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "K-NN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    # List to store results
    results = []
    pca_components_list = [10, 15, 20, 25, 30]

    # Train models with different PCA components
    for n_components in pca_components_list:
        X_train_pca, X_test_pca = apply_pca(X_train_flat, X_test_flat, n_components)
        results.extend(train_and_evaluate(models, X_train_pca, X_test_pca, Y_train, Y_test, n_components))

    # Convert results to a DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv("model_performance_analysis.csv", index=False)

    # Plot results
    plot_results(df_results)

    # Print best models based on F1-score & ROC-AUC
    best_f1_model = df_results.loc[df_results.groupby("PCA Components")["F1-score"].idxmax()]
    best_roc_auc_model = df_results.loc[df_results.groupby("PCA Components")["ROC-AUC"].idxmax()]
    print("\n Best Model Based on F1-Score:\n", best_f1_model)
    print("\nBest Model Based on ROC-AUC:\n", best_roc_auc_model)


# Run script
if __name__ == "__main__":
    main()
