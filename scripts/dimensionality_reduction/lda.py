from enrichr import enrichr_query
from omic_array import OmicArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import math
import numpy as np
import pickle


def plot_lda_coefficients(results):
    x_plot, y_plot, stds = [], [], []
    min_magnitude, max_magnitude = 0, 0
    for subtype, fields in results.items():
        if min(list(fields["features"].keys())) < min_magnitude:
            min_magnitude = min(list(fields["features"].keys()))
        if max(list(fields["features"].keys())) > max_magnitude:
            max_magnitude = max(list(fields["features"].keys()))

    for m in range(min_magnitude, max_magnitude+1):
        x_plot.append("10^{}".format(m))
        count = []
        for subtype, fields in results.items():
            if m in fields["features"]:
                count.append(len(fields["features"][m]))
            else:
                count.append(0)
        y_plot.append(np.mean(count))
        stds.append(np.std(count))

    print(x_plot)
    print(y_plot)
    print(stds)
    plt.figure()
    plt.bar(x_plot, y_plot, yerr=stds)
    plt.title("Distribution of the parameters magnitude")
    plt.show()


def magnitude_order(x):
    return int(math.log10(x))


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
    start = magnitude_order(min(np.abs(coefficients)))
    stop = magnitude_order(max(np.abs(coefficients)))
    results[start] = omic_array.get_omic_column_index().to_series().loc[np.abs(coefficients) <= 10**start].to_list()
    c = np.count_nonzero(np.abs(coefficients) <= 10**start)
    for magnitude in range(start+1, stop+1):
        condition = (10**(magnitude-1) < np.abs(coefficients)) & (np.abs(coefficients) <= 10**magnitude)
        c += np.count_nonzero(condition)
        results[magnitude] = omic_array.get_omic_column_index().to_series().loc[condition].to_list()
    return results


def lda_feature_selection(omic_array, features=None, features_magnitude=None):
    results = dict()
    for subtype in omic_array.pheno_unique_values("subtype"):
        # One-vs-All analysis using a single class
        ova = omic_array.pheno_replace(omic_array.pheno["subtype"] != subtype, "subtype", "Other", inplace=False)
        if features is not None and features_magnitude is not None:
            ova.select_features_omic(features[subtype]["features"][features_magnitude])
        # LDA fitting
        lda = LinearDiscriminantAnalysis(solver="svd")
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
            "features": coefficients_by_magnitude(lda.scalings_, ova)
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

    # The 368 cpg islands
    cpg_368 = set(open("../../data/features/368/368_cpg.txt").read().split('\n'))
    cpg_368_t = set(open("../../data/features/368/368_tumor_cpg.txt").read().split('\n'))

    # Loading details about CpG sites
    cpg_info = pickle.load(open("../../data/features/cpg_info.pkl", "rb"))

    # Identification of the CpG sites associated to the PAM50 genes
    pam50_cpg = set()
    for cpg, info in cpg_info.items():
        if len(pam50_genes.intersection(info["genes"])) > 0:
            pam50_cpg.add(cpg)

    # Loading the methylation 450k values for breast cancer
    bm_450 = OmicArray(filename="../../data/breast/methylation/breast_450k_final.pkl")

    bm_27 = OmicArray(filename="../../data/breast/methylation/breast_methylation_27k.pkl")

    common_index = bm_27.get_omic_column_index().intersection(bm_450.get_omic_column_index())
    bm_450.append(bm_27)
    bm_450.select_features_omic(common_index.to_list())

    bm_450.select_features_omic(pam50_cpg.union(cpg_368_t))

    run_1 = lda_feature_selection(bm_450)
    for k, v in run_1.items():
        print("{} - Sn: {:.2f} - Sp: {:.2f} - AUC: {:.2f}".format(k, v["sensitivity"], v["specificity"], v["auc"]))
    print()
    plot_lda_coefficients(run_1)
    highest_mag = set()
    for k, v in run_1.items():
        highest_mag = highest_mag.union(v["features"][-1])

    # plot_lda_coefficients(lda_feature_selection(bm_450))
    for k, v in lda_feature_selection(bm_450, features=run_1, features_magnitude=-1).items():
        print("{} - Sn: {:.2f} - Sp: {:.2f} - AUC: {:.2f}".format(k, v["sensitivity"], v["specificity"], v["auc"]))
    print()
    gene_highest = set()
    for c in highest_mag:
        gene_highest = gene_highest.union(cpg_info[c]["genes"])

    library = "KEGG_2019_Human"

    pathways = enrichr_query(list(gene_highest), library)
    for p in pathways[library][:15]:
        print(p[1])

    exit()
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
