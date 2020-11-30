from dataset.dimensionality_reduction import coefficients_by_magnitude
import matplotlib.pyplot as plt
from omic_array import OmicArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


def lda_feature_selection(omic_array):
    # One-vs-One lda analysis
    # LDA fitting
    lda = LinearDiscriminantAnalysis(solver="svd")
    x, y = omic_array.sklearn_conversion("subtype")
    lda.fit(x, y)

    return coefficients_by_magnitude(lda.scalings_, lda.xbar_.reshape(-1, 1), omic_array)


def roc_curve(omic_array, results, positive, negative, magnitudes=None):
    if magnitudes is None:
        magnitudes = list(results.keys())

    features, offsets, scalings = [], [], []
    for m in magnitudes:
        features += results[m]["feats"]
        offsets += list(results[m]["offsets"])
        scalings += list(results[m]["coefficients"])

    omic_array.omic = omic_array.omic[features]
    x, y = omic_array.sklearn_conversion("subtype")
    x = np.dot(x-offsets, scalings)
    thresholds = np.append(np.sort(x), np.max(x)+1)
    x_axis, y_axis = [], []
    for t in thresholds:
        y_pred = [negative if f < t else positive for f in x]
        report = classification_report(y.ravel(), y_pred, output_dict=True)
        x_axis.append(1 - report[negative]["recall"])
        y_axis.append(report[positive]["recall"])

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title("{} vs {}".format(positive, negative))
    plt.show()


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

    dataset_choice = 1
    if dataset_choice == 1:
        dataset = bm_450
    elif dataset_choice == 2:
        common_index = bm_27.get_omic_column_index().intersection(bm_450.get_omic_column_index())
        bm_450.append(bm_27)
        bm_450.select_features_omic(common_index.to_list())
        dataset = bm_450

    dataset_mod = None
    if dataset_mod == "PAM50":
        dataset.select_features_omic(pam50_cpg)
    elif dataset_mod == "368ALL":
        dataset.select_features_omic(pam50_cpg.union(cpg_368))
    elif dataset_mod == "368REV":
        dataset.select_features_omic(pam50_cpg.union(cpg_368_t))

    subtypes = dataset.pheno_unique_values("subtype")
    for i in range(len(subtypes)):
        subtype_1 = subtypes[i]
        for j in range(i+1, len(subtypes)):
            subtype_2 = subtypes[j]
            work_dataset = dataset.deep_copy()
            work_dataset.filter_classes("subtype", subtype_1, subtype_2)
            work_dataset.scale_features(StandardScaler())
            r = lda_feature_selection(work_dataset)
            roc_curve(work_dataset, r, subtype_1, subtype_2)


