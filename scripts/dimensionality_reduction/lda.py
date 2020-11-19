from enrichr import enrichr_query
from omic_array import OmicArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pickle


# plot function for LDA coefficients
def plot_lda_coefficients(lda_scalings, plt_title):
    s = np.abs(lda_scalings.ravel())
    x_plot, y_plot = list(range(-7, 3)), []
    for magnitude in range(-7, 3):
        if magnitude == -7:
            y_plot.append(np.count_nonzero(s <= 10**-7))
        elif magnitude == 2:
            y_plot.append(np.count_nonzero(s > 10**2))
        else:
            y_plot.append(np.count_nonzero((10**(magnitude-1) < s) & (s <= 10**magnitude)))

    plt.figure()
    plt.bar(x_plot, y_plot)
    plt.title(plt_title)
    plt.show()


def train_classifier(x, y, positive_class):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    classifier = SVC()
    classifier.fit(x_train, y_train)
    y_test = [1 if el == positive_class else 0 for el in y_test]
    y_pred = classifier.predict(x_test)
    y_pred = [1 if el == positive_class else 0 for el in y_pred]
    report = classification_report(y_test, y_pred, output_dict=True)
    sensitivity = report["1"]["recall"]
    specificity = report["0"]["recall"]
    auc = roc_auc_score(y_test, y_pred)
    return sensitivity, specificity, auc


def coefficients_by_magnitude(coefficients, omic_array):
    results = dict()
    results[-7] = omic_array.get_omic_column_index().to_series().loc[np.abs(coefficients) <= 10**-7].to_list()
    results[0] = omic_array.get_omic_column_index().to_series().loc[np.abs(coefficients) > 1].to_list()
    for magnitude in range(-6, 0):
        condition = (10**(magnitude-1) < np.abs(coefficients)) & (np.abs(coefficients) <= 10**magnitude)
        results[magnitude] = omic_array.get_omic_column_index().to_series().loc[condition].to_list()
    return results


def lda_feature_selection(omic_array, shrinkage=None):
    results = dict()
    for subtype in omic_array.pheno_unique_values("subtype"):
        # One-vs-All analysis using a single class
        ova = omic_array.pheno_replace(omic_array.pheno["subtype"] != subtype, "subtype", "Other", inplace=False)
        # LDA fitting
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage)
        x, y = ova.sklearn_conversion("subtype")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x = lda.fit_transform(x, y)
        # A classifier is trained using the reduced-dimensionality representation of the data
        sn, sp, auc = train_classifier(x, y, subtype)
        results[subtype] = {
            "sensitivity": sn,
            "specificity": sp,
            "auc": auc,
            "features": coefficients_by_magnitude(lda.scalings_, omic_array)
        }
    return results


def sgd_feature_selection(omic_array, l2_regularization=0.0001):
    results = dict()
    for subtype in omic_array.pheno_unique_values("subtype"):
        # One-vs-All analysis using a single class
        ova = omic_array.pheno_replace(omic_array.pheno["subtype"] != subtype, "subtype", "Other", inplace=False)

        sgd = SGDClassifier(loss="hinge", alpha=l2_regularization)
        x, y = ova.sklearn_conversion("subtype")
        # Train/test splitting
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Fitting
        sgd.fit(x_test, y_test)
        y_test = [1 if el == subtype else 0 for el in y_test]
        y_pred = sgd.predict(x_test)
        y_pred = [1 if el == subtype else 0 for el in y_pred]
        report = classification_report(y_test, y_pred, output_dict=True)
        sensitivity = report["1"]["recall"]
        specificity = report["0"]["recall"]
        auc = roc_auc_score(y_test, y_pred)

        results[subtype] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc,
            "features": coefficients_by_magnitude(sgd.coef_[0], omic_array)
        }
    return results


if __name__ == "__main__":
    # OBJECTIVE: feature extraction using LDA

    # The genes from the PAM50 panel
    pam50_genes = set(open("../../data/features/PAM50/PAM50_genes.txt").read().split('\n'))

    # Loading details about CpG sites
    cpg_info = pickle.load(open("../../data/features/cpg_info.pkl", "rb"))

    # Identification of the CpG sites associated to the PAM50 genes
    pam50_cpg = set()
    for cpg, info in cpg_info.items():
        if len(pam50_genes.intersection(info["genes"])) > 0:
            pam50_cpg.add(cpg)

    # Loading the methylation 450k values for breast cancer
    bm_450 = OmicArray(filename="../../data/breast/methylation/breast_450k_final.pkl")
    # bm_450.select_features_omic(pam50_cpg)

    # LDA example using different shrinkage values
    shrinkage_results = dict()
    for i in np.linspace(10**-4, 1, 50):
        shrinkage_results[i] = sgd_feature_selection(bm_450, i)
    # plotting the results
    fig, axs = plt.subplots(2, 1)
    x_axis = list(shrinkage_results.keys())
    for s in bm_450.pheno_unique_values("subtype"):
        y_1, y_2 = [], []
        for k, v in shrinkage_results.items():
            y_1.append(v[s]["auc"])
            discarded_features = 0
            for i in range(-7, -1):
                discarded_features += len(v[s]["features"][i])
            y_2.append(discarded_features)
        axs[0].plot(x_axis, y_1, label=s)
        axs[1].plot(x_axis, y_2, label=s)
    axs[0].set_title("AUC varying regularization on LDA")
    axs[0].set_xlabel("Regularization parameter")
    axs[0].set_ylabel("AUC")

    axs[1].set_title("# features with coefficients > 0.1 on LDA")
    axs[1].set_xlabel("Regularization parameter")
    axs[1].set_ylabel("# features")
    plt.show()
