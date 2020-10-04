from methylnet_utils import merge_methylation_arrays, split_methylation_array_by_pheno
from models.autoencoders import Giskard
from models.classifiers import Daneel, Jander
from models.generators import AutoencoderGenerator, MethylationArrayGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


# SOME FUNCTIONS
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
              500,
              verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20)])
    test_accuracy = model.evaluate(test_data["beta"].to_numpy(),
                                   pd.get_dummies(test_data["pheno"]["subtype"]).to_numpy())
    return model, test_accuracy


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


# Defining parameters for training
omics = ["methylation", "mrna", "mirna", "combined"]
methylation_latent_dimension = 200
mirna_latent_dimension = 200
mrna_latent_dimension = 200
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
# mRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)
pam50_mrnas = list()
for m in pam50_mirnas:
    for gene in mirna_to_genes[m]:
        if gene[2] > 0.8:
            pam50_mrnas.append(gene[0])
# miRNA selection
pam50_mirnas = list()
for mirna, genes in mirna_to_genes.items():
    for gene in genes:
        if gene[1] in pam50_genes and gene[2] > 0.8:
            pam50_mirnas.append(mirna)

# PART 2
# Training the different autoencoders
# Training the methylation autoencoder
datasets = dict()
methylation_dataset = pickle.load(open("../data/final_preprocessed/breast_methylation_450_ma.pkl", "rb"))
index = set(methylation_dataset["beta"].columns.values)
pam50_cpgs = index.intersection(pam50_cpgs)
methylation_dataset["beta"] = methylation_dataset["beta"][pam50_cpgs]
methylation_encoder = train_autoencoder(methylation_dataset, methylation_latent_dimension)
datasets["methylation"] = {
    "original": correct_labels(methylation_dataset),
    "embedded": methylation_encoder.encode_methylation_array(methylation_dataset)
}
# Training the mRNA autoencoder
mrna_dataset = pickle.load(open("../data/mrna_exp_ma.pkl", "rb"))
mrna_dataset["beta"] = mrna_dataset["beta"].rename(columns=lambda g: g.split('.')[0])
mrna_dataset["beta"] = mrna_dataset["beta"][pam50_mrnas]
mrna_encoder = train_autoencoder(mrna_dataset, mrna_latent_dimension)
datasets["mrna"] = {
    "original": correct_labels(mrna_dataset),
    "embedded": mrna_encoder.encode_methylation_array(mrna_dataset)
}
# Training the miRNA autoencoder
mirna_dataset = pickle.load(open("../data/mirna_exp_ma.pkl", "rb"))
mirna_dataset["beta"] = mirna_dataset["beta"][pam50_mirnas]
mirna_encoder = train_autoencoder(mirna_dataset, mirna_latent_dimension)
datasets["mirna"] = {
    "original": correct_labels(mirna_dataset),
    "embedded": mirna_encoder.encode_methylation_array(mirna_dataset)
}
# Combined dataset
datasets["combined"] = {
    "embedded": merge_methylation_arrays(
        datasets["methylation"]["embedded"],
        datasets["mrna"]["embedded"],
        datasets["mirna"]["embedded"]
    )
}

encoders = {
    "methylation": methylation_encoder,
    "mrna": mrna_encoder,
    "mirna": mirna_encoder
}

# PART 3
# Training the different classifiers
stats = {
    "base": {
        "methylation": {
            "jander": [],
            "daneel": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "mrna": {
            "jander": [],
            "daneel": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "mirna": {
            "jander": [],
            "daneel": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
        "combined": {
            "jander": [],
            "daneel": [],
            "svm": [],
            "knn": [],
            "rf": []
        },
    }
}

barcodes, models = dict(), dict(),
for i in range(10):
    for omic in omics:
        training_set, validation_set, test_set = \
            split_methylation_array_by_pheno(datasets[omic]["embedded"], "subtype", val_rate=0.1, test_rate=0.1)
        barcodes[omic] = list(test_set["pheno"].index.values)
        models[omic] = dict()

        # DNN - Jander
        models[omic]["jander"], acc = train_dnn_classifier(Jander, training_set, validation_set, test_set)
        stats["base"][omic]["jander"].append(acc)

        # DNN - Daneel
        models[omic]["daneel"], acc = train_dnn_classifier(Daneel, training_set, validation_set, test_set)
        stats["base"][omic]["daneel"].append(acc)

        # SVM
        models[omic]["svm"], acc = train_ml_classifier(training_set, test_set, SVC, {"C": 1, "kernel": "rbf"},
                                                       validation_set)
        stats["base"][omic]["svm"].append(acc)

        # KNN
        models[omic]["knn"], acc = train_ml_classifier(training_set, test_set, KNeighborsClassifier,
                                                       {"n_neighbors": 75}, validation_set)
        stats["base"][omic]["knn"].append(acc)

        # RF
        models[omic]["rf"], acc = train_ml_classifier(training_set, test_set, RandomForestClassifier,
                                                      {"n_estimators": 2000, "max_features": "auto"}, validation_set)
        stats["base"][omic]["rf"].append(acc)

datasets["combined"]["original"] = {
    "methylation": slice_methylation_array_to_merge(datasets["methylation"]["original"], barcodes["combined"]),
    "mrna": slice_methylation_array_to_merge(datasets["mrna"]["original"], barcodes["combined"]),
    "mirna": slice_methylation_array_to_merge(datasets["mirna"]["original"], barcodes["combined"])
}

for o, omic_stats in stats["base"].items():
    for c, scores in omic_stats.items():
        print("{} {} mean acc: {} - std {}".format(o, c, np.mean(scores), np.std(scores)))

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

# Effect of noise on single genes
noise_on_genes = dict()
index = set(methylation_dataset["beta"].columns.values)
for gene in pam50_genes:
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
