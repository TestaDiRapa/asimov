from methylnet_utils import balanced_kcv, merge_methylation_arrays
from models.autoencoders import Giskard
from models.classifiers import Daneel, Jander
from models.generators import AutoencoderGenerator, MethylationArrayGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

tf.get_logger().setLevel('ERROR')


# SOME FUNCTIONS
def filter_and_replace(methylation_array, barcodes, dummies=True):
    p = methylation_array["pheno"].replace(["LuminalB", "LuminalA", "control", "TNBS", "Unclear", "HER2+"],
                                           ["Luminal B", "Luminal A", "control", "Basal-like", "NA", "HER2-enriched"])
    if dummies:
        p = pd.get_dummies(p["subtype"])
    return {
        "beta": methylation_array["beta"].drop(barcodes),
        "pheno": p.drop(barcodes)
    }


def ma_loc(methylation_array, barcodes):
    return {
        "beta": methylation_array["beta"].loc[barcodes],
        "pheno": methylation_array["pheno"].loc[barcodes]
    }


def print_class_count(df):
    count_tmp = dict()
    for el in df["pheno"]["subtype"].unique():
        count_tmp[el] = 0

    for index, row in df["pheno"].iterrows():
        count_tmp[row["subtype"]] += 1
    print(count_tmp)


def train_autoencoder(methylation_array, latent_dimension):
    val_size = int(methylation_array["beta"].shape[0] * 0.1)
    val_set = AutoencoderGenerator(methylation_array["beta"].iloc[:val_size, :])
    train_set = AutoencoderGenerator(methylation_array["beta"].iloc[val_size:, :])

    # Autoencoder training
    encoder = Giskard(methylation_array["beta"].shape[1], latent_dimension=latent_dimension,
                      model_serialization_path="../data/models/")
    encoder.fit(train_set, val_set, 500,
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
    return encoder


def train_autoencoder_all(methylation_array, latent_dimension):
    val_size = int(methylation_array.shape[0] * 0.1)
    val_set = AutoencoderGenerator(methylation_array.iloc[:val_size, :])
    train_set = AutoencoderGenerator(methylation_array.iloc[val_size:, :])

    # Autoencoder training
    encoder = Giskard(methylation_array.shape[1], latent_dimension=latent_dimension,
                      model_serialization_path="../data/models/")
    encoder.fit(train_set, val_set, 500,
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10)])
    return encoder


def correct_labels(methylation_array):
    gt_check = pd.read_csv("../data/brca_tcga_pub_clinical_data.tsv", sep="\t", na_filter=False, index_col="Patient ID")
    gt_index = list(gt_check.index.values)
    to_remove = list()
    for pheno_index, row in methylation_array["pheno"].iterrows():
        if row["subtype"] != "control":
            barcode = "-".join(pheno_index.split("-")[:3])
            if barcode in gt_index and gt_check.loc[barcode]["PAM50 subtype"] != "Normal-like":
                methylation_array["pheno"].at[pheno_index, "subtype"] = gt_check.loc[barcode]["PAM50 subtype"]
            else:
                to_remove.append(pheno_index)

    methylation_array["beta"] = methylation_array["beta"].drop(to_remove)
    methylation_array["pheno"] = methylation_array["pheno"].drop(to_remove)
    return methylation_array


def train_dnn_classifier(classifier, train_set, val_set, test_data):
    params = {"input_shape": train_set["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
              "dropout_rate": 0.3, "output_shape": len(train_set["pheno"]["subtype"].unique())}
    model = classifier(**params)
    model.fit(MethylationArrayGenerator(train_set, "subtype"),
              MethylationArrayGenerator(val_set, "subtype"),
              40,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])
    test_accuracy = model.evaluate(test_data["beta"].to_numpy(),
                                   pd.get_dummies(test_data["pheno"]["subtype"]).to_numpy())
    return model, test_accuracy


def train_dnn_classifier_deep_cc(classifier, train_set, val_set, test_data):
    params = {"input_shape": train_set["beta"].shape[1], "model_serialization_path": "../data/models/classifier/",
              "dropout_rate": 0.3, "output_shape": len(train_set["pheno"]["subtype"].unique())}
    model = classifier(**params)
    model.fit(MethylationArrayGenerator(train_set, "subtype"),
              MethylationArrayGenerator(val_set, "subtype"),
              40,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])
    dummies = pd.get_dummies(train_set["pheno"]["subtype"]).columns.to_list()
    tmp_acc = []
    for p, r in zip(model.predict(test_data["beta"].to_numpy()), test_data["pheno"].values.ravel()):
        if dummies[np.argmax(p)] == r:
            tmp_acc.append(1)
        else:
            tmp_acc.append(0)
    return model, np.mean(tmp_acc)


def train_ml_classifier(train_set, test_data, model, params, val_set=None):
    if val_set is not None:
        train_set["beta"] = train_set["beta"].append(val_set["beta"])
        train_set["pheno"] = train_set["pheno"].append(val_set["pheno"])
    classifier = model(**params)
    classifier.fit(train_set["beta"], train_set["pheno"].values.ravel())
    return classifier, classifier.score(test_data["beta"], test_data["pheno"].values.ravel())


def slice_methylation_array_to_merge(methylation_array, b):
    return {
        "pheno": methylation_array["pheno"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4])).loc[b],
        "beta": methylation_array["beta"].rename(index=lambda barcode: "-".join(barcode.split("-")[:4])).loc[b]
    }


def plot_single_omic_change(acc_scores, ml_method, omic_type, single=True):
    x, y = [], []
    x.append(0)
    y.append(np.mean(acc_scores["base"][omic_type][ml_method]))
    if single:
        scope = "single"
    else:
        scope = "combined"
    for std, a in acc_scores[scope][omic_type][ml_method].items():
        x.append(std)
        y.append(a)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.title("Accuracy of {} adding noise to {}".format(ml_method, omic_type))
    plt.xlabel("Variance of the noise")
    plt.ylabel("Accuracy")
    plt.savefig("../data/results/{}_{}_{}.png".format(ml_method, omic_type, scope))


def plot_against_combined(acc_scores, ml_method, omic_type):
    x_single, y_single = [], []
    x_combined, y_combined = [], []
    x_single.append(0)
    y_single.append(np.mean(acc_scores["base"][omic_type][ml_method]))
    for std, a in acc_scores["single"][omic_type][ml_method].items():
        x_single.append(std)
        y_single.append(a)

    x_combined.append(0)
    y_combined.append(np.mean(acc_scores["base"]["combined"][ml_method]))
    for std, a in acc_scores["combined"][omic_type][ml_method].items():
        x_combined.append(std)
        y_combined.append(a)

    plt.figure(figsize=(10, 10))
    plt.plot(x_single, y_single, label="single")
    plt.plot(x_combined, y_combined, label="combined")
    plt.title("Accuracy of {} adding noise to {}".format(ml_method, omic_type))
    plt.xlabel("Variance of the noise")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("../data/results/{}_{}_single_vs_combined.png".format(ml_method, omic_type))


def remove_classes(methylation_array, classes):
    pheno = methylation_array["pheno"]
    to_remove = list(pheno[pheno["subtype"].isin(classes)].index.values)
    print("REMOVING", len(to_remove))
    return {
        "beta": methylation_array["beta"].drop(to_remove),
        "pheno": methylation_array["pheno"].drop(to_remove)
    }


def spec_and_sens(stats, cm, omic, method):
    for c in cm.index.values:
        if c not in stats["sp"][omic][method]:
            stats["sp"][omic][method][c] = []
        if c not in stats["sn"][omic][method]:
            stats["sn"][omic][method][c] = []
        tp = cm.loc[c, c]
        fp = cm.loc[:, c].sum() - tp
        fn = cm.loc[c, :].sum() - tp
        tn = cm.sum().sum() - tp - fp - fn

        stats["sp"][omic][method][c].append(tn/(tn+fp))
        stats["sn"][omic][method][c].append(tp / (tp + fn))


def ml_confusion_matrix(model, ma, target):
    columns = list(ma["pheno"][target].unique())
    confusion_matrix = pd.DataFrame(data=np.zeros((len(columns), len(columns))), columns=columns, index=columns)
    for r, p in zip(ma["pheno"].values.ravel(), model.predict(ma["beta"])):
        confusion_matrix.at[r, p] += 1
    return confusion_matrix


# Defining parameters for training
omics = ["methylation", "mrna", "mirna", "combined"]
methods = ["jander", "daneel", "svm", "knn", "rf"]
methylation_latent_dimension = 20
mirna_latent_dimension = 20
mrna_latent_dimension = 20
dataset_list = []

# PART 1
# CpG, mRNA and miRNA selection using PAM50 genes and first interactors
pam50_genes = open("../data/PAM50_genes.txt").read().split('\n')
pam50_ENSG = open("../data/PAM50_ENSG.txt").read().split('\n')
gene_to_cpgs = pickle.load(open("../data/genes_cpg_interaction.pkl", "rb"))
mirna_to_genes = pickle.load(open("../data/mirna_genes_interaction.pkl", "rb"))

# CpGs selection
pam50_cpgs = set()
for gene in pam50_genes:
    if gene in gene_to_cpgs:
        pam50_cpgs = pam50_cpgs.union(gene_to_cpgs[gene])
# miRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)
# mRNA selection
pam50_mrnas = list()
for m in pam50_mirnas:
    for gene in mirna_to_genes[m]:
        if gene[2] > 0.8:
            pam50_mrnas.append(gene[0])

pam50_cpgs = pickle.load(open("../data/cpg_list_27k.pkl", "rb"))

# PART 2
# Training the different autoencoders
# Training the methylation autoencoder
datasets = dict()
methylation_dataset_450k = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index = set(methylation_dataset_450k["beta"].columns.values)
pam50_cpgs = index.intersection(pam50_cpgs)
methylation_dataset_450k["beta"] = methylation_dataset_450k["beta"][pam50_cpgs]
# methylation_dataset_27k = pickle.load(open("../data/breast_methylation_27k.pkl", "rb"))
# methylation_dataset = dict()
# methylation_dataset["beta"] = methylation_dataset_27k["beta"].append(methylation_dataset_450k["beta"])
# methylation_dataset["pheno"] = methylation_dataset_27k["pheno"].append(methylation_dataset_450k["pheno"])
methylation_dataset = methylation_dataset_450k
print(methylation_dataset)
# encoder_dataset = pickle.load(open("../data/methylation_exp_all.pkl", "rb"))
# methylation_encoder = train_autoencoder_all(encoder_dataset, methylation_latent_dimension)
methylation_encoder = train_autoencoder(methylation_dataset, methylation_latent_dimension)
methylation_corrected = correct_labels(methylation_dataset)
methylation_tmp = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
methylation_tmp["beta"] = methylation_tmp["beta"][pam50_cpgs]
# methylation_corrected = remove_classes(methylation_corrected, ["NA", "control"])
datasets["methylation"] = {
    "original": methylation_corrected,
    # "embedded": methylation_encoder.encode_methylation_array(methylation_corrected)
    "embedded": methylation_corrected
}
# Training the mRNA autoencoder
mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])
index = set(mrna_dataset["beta"].columns.values)
pam50_mrnas = index.intersection(pam50_mrnas)
mrna_dataset["beta"] = mrna_dataset["beta"][pam50_mrnas]
# encoder_dataset = pickle.load(open("../data/mrna_exp_all.pkl", "rb"))
# encoder_dataset = encoder_dataset.rename(columns=lambda g: g.split('.')[0])
# encoder_dataset = encoder_dataset[pam50_mrnas]
# mrna_encoder = train_autoencoder_all(encoder_dataset, mrna_latent_dimension)
mrna_encoder = train_autoencoder(mrna_dataset, mrna_latent_dimension)
mrna_corrected = correct_labels(mrna_dataset)
# mrna_corrected = remove_classes(mrna_corrected, ["NA", "control"])
mrna_tmp = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_tmp["beta"] = mrna_tmp["beta"].rename(columns=lambda g: g.split('.')[0])[pam50_mrnas]
datasets["mrna"] = {
    "original": mrna_corrected,
    "embedded": mrna_encoder.encode_methylation_array(mrna_corrected)
}
# Training the miRNA autoencoder
mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
# encoder_dataset = pickle.load(open("../data/mirna_exp_all.pkl", "rb"))
# encoder_dataset = encoder_dataset[pam50_mirnas]
# mirna_encoder = train_autoencoder_all(encoder_dataset, mirna_latent_dimension)
mirna_encoder = train_autoencoder(mirna_dataset, mirna_latent_dimension)
mirna_corrected = correct_labels(mirna_dataset)
# mirna_corrected = remove_classes(mirna_corrected, ["NA", "control"])
mirna_tmp = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_tmp["beta"] = mirna_tmp["beta"][pam50_mirnas]
datasets["mirna"] = {
    "original": mirna_corrected,
    "embedded": mirna_encoder.encode_methylation_array(mirna_corrected)
}
# Combined dataset
datasets["combined"] = {
    "embedded": merge_methylation_arrays(
        datasets["methylation"]["embedded"],
        datasets["mrna"]["embedded"],
        datasets["mirna"]["embedded"]
    )
}

for omic in omics:
    print_class_count(datasets[omic]["embedded"])

encoders = {
    "methylation": methylation_encoder,
    "mrna": mrna_encoder,
    "mirna": mirna_encoder
}

# PART 3
# Training the different classifiers
stats = dict()
stats["base"] = dict()
stats["sp"] = dict()
stats["sn"] = dict()
for omic in omics:
    stats["base"][omic] = dict()
    stats["sp"][omic] = dict()
    stats["sn"][omic] = dict()
    for m in methods:
        stats["base"][omic][m] = list()
        stats["sp"][omic][m] = dict()
        stats["sn"][omic][m] = dict()

barcodes, models = dict(), dict()
k = 10

for omic in omics:
    kcv_barcodes = balanced_kcv(datasets[omic]["embedded"], "subtype", k)
    for i in range(k):
        val_index = i
        test_index = i+1
        if test_index == k:
            test_index = 0
        validation_set = ma_loc(datasets[omic]["embedded"], kcv_barcodes[val_index])
        test_set = ma_loc(datasets[omic]["embedded"], kcv_barcodes[test_index])
        training_barcodes = []
        for j in range(k):
            if j != val_index and j != test_index:
                training_barcodes += kcv_barcodes[j]
        training_set = ma_loc(datasets[omic]["embedded"], training_barcodes)
        barcodes[omic] = list(test_set["pheno"].index.values)
        models[omic] = dict()

        # DNN - Jander
        models[omic]["jander"], acc = train_dnn_classifier(Jander, training_set, validation_set, test_set)
        stats["base"][omic]["jander"].append(acc)
        spec_and_sens(stats,
                      models[omic]["jander"].confusion_matrix(test_set["beta"].to_numpy(), test_set["pheno"], "subtype"),
                      omic,
                      "jander")

        # DNN - Daneel
        models[omic]["daneel"], acc = train_dnn_classifier(Daneel, training_set, validation_set, test_set)
        stats["base"][omic]["daneel"].append(acc)
        spec_and_sens(stats,
                      models[omic]["jander"].confusion_matrix(test_set["beta"].to_numpy(), test_set["pheno"], "subtype"),
                      omic,
                      "daneel")

        # SVM
        models[omic]["svm"], acc = train_ml_classifier(training_set, test_set, SVC, {"C": 1, "kernel": "rbf"},
                                                       validation_set)
        stats["base"][omic]["svm"].append(acc)
        spec_and_sens(stats,
                      ml_confusion_matrix(models[omic]["svm"], test_set, "subtype"),
                      omic,
                      "svm")

        # KNN
        models[omic]["knn"], acc = train_ml_classifier(training_set, test_set, KNeighborsClassifier,
                                                       {"n_neighbors": 75}, validation_set)
        stats["base"][omic]["knn"].append(acc)
        spec_and_sens(stats,
                      ml_confusion_matrix(models[omic]["knn"], test_set, "subtype"),
                      omic,
                      "knn")

        # RF
        models[omic]["rf"], acc = train_ml_classifier(training_set, test_set, RandomForestClassifier,
                                                      {"n_estimators": 2000, "max_features": "auto"}, validation_set)
        stats["base"][omic]["rf"].append(acc)
        spec_and_sens(stats,
                      ml_confusion_matrix(models[omic]["rf"], test_set, "subtype"),
                      omic,
                      "rf")

datasets["combined"]["original"] = {
    "methylation": slice_methylation_array_to_merge(datasets["methylation"]["original"], barcodes["combined"]),
    "mrna": slice_methylation_array_to_merge(datasets["mrna"]["original"], barcodes["combined"]),
    "mirna": slice_methylation_array_to_merge(datasets["mirna"]["original"], barcodes["combined"])
}

for o, omic_stats in stats["base"].items():
    for c, scores in omic_stats.items():
        print("{} {} mean acc: {:.3f} - std {:.3f}".format(o, c, np.mean(scores), np.std(scores)))
print()

for o in omics:
    print(o)
    for m in methods:
        print(m)
        for c in stats["sp"][o][m].keys():
            print("{} - sp: {:.3f} - sn {:.3f}".format(c, np.mean(stats["sp"][o][m][c]), np.mean(stats["sn"][o][m][c])))
        print()

deep_cc_brca = pd.read_csv("../data/deep_cc_brca.csv", index_col=1).rename(index=lambda b: "-".join(b.split("-")[:4]))
deep_cc_brca = deep_cc_brca.dropna()
deep_cc_brca = deep_cc_brca.replace(["LumB", "LumA", "control", "Basal", "NA", "Her2"],
                                    ["Luminal B", "Luminal A", "control", "Basal-like", "NA", "HER2-enriched"])
deep_cc_brca = deep_cc_brca.loc[:, ["DeepCC"]].rename(columns={"DeepCC": "subtype"})

for omic in omics:
    datasets[omic]["embedded"]["pheno"].rename(index=lambda b: "-".join(b.split("-")[:4]), inplace=True)
    datasets[omic]["embedded"]["beta"].rename(index=lambda b: "-".join(b.split("-")[:4]), inplace=True)

ref_index = set(datasets["combined"]["embedded"]["pheno"].index.values)
cc_index = set(deep_cc_brca.index.values)
deep_cc_brca = deep_cc_brca.loc[cc_index.intersection(ref_index)]
print(deep_cc_brca)
deep_cc_kcv = balanced_kcv({"beta": pd.DataFrame(), "pheno": deep_cc_brca}, "subtype", k)

stats_cc = dict()
stats_cc["base"] = dict()
stats_cc["sp"] = dict()
stats_cc["sn"] = dict()
for omic in omics:
    stats_cc["base"][omic] = dict()
    stats_cc["sp"][omic] = dict()
    stats_cc["sn"][omic] = dict()
    for m in methods:
        stats_cc["base"][omic][m] = list()
        stats_cc["sp"][omic][m] = dict()
        stats_cc["sn"][omic][m] = dict()

for omic in omics:
    models[omic] = dict()
    for cc_barcodes in deep_cc_kcv:
        test_set = ma_loc(datasets[omic]["embedded"], cc_barcodes)
        for b in test_set["pheno"].index.values:
            test_set["pheno"].at[b,"subtype"] = deep_cc_brca.loc[b, "subtype"]
        new_dataset = {
            "beta": datasets[omic]["embedded"]["beta"].drop(cc_barcodes),
            "pheno": datasets[omic]["embedded"]["pheno"].drop(cc_barcodes)
        }

        kcv_barcodes = balanced_kcv(new_dataset, "subtype", k)
        validation_set = ma_loc(new_dataset, kcv_barcodes[0])
        training_barcodes = []
        for j in range(1, k):
            training_barcodes += kcv_barcodes[j]
        training_set = ma_loc(new_dataset, training_barcodes)

        # DNN - Jander
        models[omic]["jander"], acc = train_dnn_classifier_deep_cc(Jander, training_set, validation_set, test_set)
        stats_cc["base"][omic]["jander"].append(acc)
        spec_and_sens(stats_cc,
                      models[omic]["jander"].confusion_matrix(test_set["beta"].to_numpy(), test_set["pheno"], "subtype"),
                      omic,
                      "jander")

        # DNN - Daneel
        models[omic]["daneel"], acc = train_dnn_classifier_deep_cc(Daneel, training_set, validation_set, test_set)
        stats_cc["base"][omic]["daneel"].append(acc)
        spec_and_sens(stats_cc,
                      models[omic]["jander"].confusion_matrix(test_set["beta"].to_numpy(), test_set["pheno"], "subtype"),
                      omic,
                      "daneel")

        # SVM
        models[omic]["svm"], acc = train_ml_classifier(training_set, test_set, SVC, {"C": 1, "kernel": "rbf"},
                                                       validation_set)
        stats_cc["base"][omic]["svm"].append(acc)
        spec_and_sens(stats_cc,
                      ml_confusion_matrix(models[omic]["svm"], test_set, "subtype"),
                      omic,
                      "svm")

        # KNN
        models[omic]["knn"], acc = train_ml_classifier(training_set, test_set, KNeighborsClassifier,
                                                       {"n_neighbors": 75}, validation_set)
        stats_cc["base"][omic]["knn"].append(acc)
        spec_and_sens(stats_cc,
                      ml_confusion_matrix(models[omic]["knn"], test_set, "subtype"),
                      omic,
                      "knn")

        # RF
        models[omic]["rf"], acc = train_ml_classifier(training_set, test_set, RandomForestClassifier,
                                                      {"n_estimators": 2000, "max_features": "auto"}, validation_set)
        stats_cc["base"][omic]["rf"].append(acc)
        spec_and_sens(stats_cc,
                      ml_confusion_matrix(models[omic]["rf"], test_set, "subtype"),
                      omic,
                      "rf")

for o, omic_stats in stats_cc["base"].items():
    for c, scores in omic_stats.items():
        print("{} {} mean acc: {:.3f} - std {:.3f}".format(o, c, np.mean(scores), np.std(scores)))

for o in omics:
    print(o)
    for m in methods:
        print(m)
        for c in stats_cc["sp"][o][m].keys():
            print("{} - sp: {:.3f} - sn {:.3f}".format(c, np.mean(stats_cc["sp"][o][m][c]),
                                                       np.mean(stats_cc["sn"][o][m][c])))
        print()

raise(Exception())
beta = datasets["combined"]["embedded"]["beta"].loc[barcodes["combined"]]
pheno = datasets["combined"]["embedded"]["pheno"].loc[barcodes["combined"]]
new_dataset = {"beta": beta, "pheno": pheno}
cf = models["combined"]["daneel"].confusion_matrix(new_dataset["beta"].to_numpy(), new_dataset["pheno"], "subtype")


stats["single"] = {
    "methylation": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    },
    "mrna": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    },
    "mirna": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    }
}
stats["combined"] = {
    "methylation": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    },
    "mrna": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    },
    "mirna": {
        "jander": dict(),
        "daneel": dict(),
        "svm": dict(),
        "knn": dict(),
        "rf": dict()
    }
}

# Effect of noise on data
for omic in ["methylation", "mrna", "mirna"]:
    for sigma in np.linspace(0.01, 100, num=100):
        beta = datasets[omic]["original"]["beta"].loc[barcodes[omic]]
        beta = beta + np.random.normal(0, sigma, beta.shape)
        pheno = datasets[omic]["original"]["pheno"].loc[barcodes[omic]]
        new_dataset = {"beta": beta, "pheno": pheno}
        new_embedded = encoders[omic].encode_methylation_array(new_dataset)
        stats["single"][omic]["jander"][sigma] = models[omic]["jander"]\
            .evaluate(new_embedded["beta"].to_numpy(), pd.get_dummies(new_embedded["pheno"]["subtype"]).to_numpy())
        stats["single"][omic]["daneel"][sigma] = models[omic]["daneel"]\
            .evaluate(new_embedded["beta"].to_numpy(), pd.get_dummies(new_embedded["pheno"]["subtype"]).to_numpy())
        stats["single"][omic]["svm"][sigma] = models[omic]["svm"].score(new_embedded["beta"],
                                                                        new_embedded["pheno"].values.ravel())
        stats["single"][omic]["knn"][sigma] = models[omic]["knn"].score(new_embedded["beta"],
                                                                        new_embedded["pheno"].values.ravel())
        stats["single"][omic]["rf"][sigma] = models[omic]["rf"].score(new_embedded["beta"],
                                                                      new_embedded["pheno"].values.ravel())

        to_merge = []
        for o in ["methylation", "mrna", "mirna"]:
            if o != omic:
                to_merge.append(encoders[o].encode_methylation_array(datasets["combined"]["original"][o]))
            else:
                beta = datasets["combined"]["original"][o]["beta"].iloc[:, :]
                beta = beta + np.random.normal(0, sigma, beta.shape)
                to_merge.append(encoders[o].encode_methylation_array({
                    "beta": beta,
                    "pheno": datasets["combined"]["original"][o]["pheno"]
                }))

        new_embedded = merge_methylation_arrays(*to_merge)
        stats["combined"][omic]["jander"][sigma] = models["combined"]["jander"] \
            .evaluate(new_embedded["beta"].to_numpy(), pd.get_dummies(new_embedded["pheno"]["subtype"]).to_numpy())
        stats["combined"][omic]["daneel"][sigma] = models["combined"]["daneel"] \
            .evaluate(new_embedded["beta"].to_numpy(), pd.get_dummies(new_embedded["pheno"]["subtype"]).to_numpy())
        stats["combined"][omic]["svm"][sigma] = \
            models["combined"]["svm"].score(new_embedded["beta"], new_embedded["pheno"].values.ravel())
        stats["combined"][omic]["knn"][sigma] = \
            models["combined"]["knn"].score(new_embedded["beta"], new_embedded["pheno"].values.ravel())
        stats["combined"][omic]["rf"][sigma] = \
            models["combined"]["rf"].score(new_embedded["beta"], new_embedded["pheno"].values.ravel())

for omic in ["methylation", "mrna", "mirna"]:
    for ml in ["jander", "daneel", "svm", "knn", "rf"]:
        plot_against_combined(stats, ml, omic)

raise(Exception(""))
# Effect of noise on single genes
noise_on_genes = dict()
index = set(methylation_dataset["beta"].columns.values)
for gene in pam50_genes:
    print(gene)
    if gene in gene_to_cpgs:
        noise_on_genes[gene] = list()
        for i in range(100):
            new_noise = list()
            for sigma in np.linspace(0.001, 100, num=50):
                beta = datasets["methylation"]["original"]["beta"].loc[barcodes["methylation"]]
                noise = np.zeros(beta.shape)
                gene_cpgs = gene_to_cpgs[gene].intersection(index)
                for cpg in gene_cpgs:
                    cpg_col = beta.columns.get_loc(cpg)
                    noise[:, cpg_col] = np.random.normal(0, sigma, beta.shape[0])
                beta = beta + noise
                pheno = datasets["methylation"]["original"]["pheno"].loc[barcodes["methylation"]]
                new_dataset = {"beta": beta, "pheno": pheno}
                new_embedded = encoders["methylation"].encode_methylation_array(new_dataset)
                new_noise.append(models["methylation"]["daneel"].evaluate(
                    new_embedded["beta"].to_numpy(),
                    pd.get_dummies(new_embedded["pheno"]["subtype"]).to_numpy()))
            noise_on_genes[gene].append(new_noise)
pickle.dump(noise_on_genes, open("../data/noise_effect_on_genes.pkl", "wb"))
