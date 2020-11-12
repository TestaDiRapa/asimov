from omic_array import OmicArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# plot function for LDA coefficients
def plot_lda_coefficients(lda_scalings):
    s = np.abs(lda_scalings.ravel())
    x_plot, y_plot = list(range(-7, 3)), []
    for magnitude in range(-7, 3):
        if magnitude == -7:
            y_plot.append(np.count_nonzero(s <= 10**-7))
        elif magnitude == 2:
            y_plot.append(np.count_nonzero(s > 10**2))
        else:
            y_plot.append(np.count_nonzero((10**(magnitude-1) < s) & (s <= 10**magnitude)))

    plt.bar(x_plot, y_plot)
    plt.show()


# OBJECTIVE: feature extraction using LDA

# Loading the methylation 450k values for breast cancer
bm_450 = OmicArray(filename="../../data/breast/methylation/breast_450k_final.pkl")

for subtype in bm_450.pheno_unique_values("subtype"):
    # One-vs-All analysis using a single class
    bm_ova = bm_450.pheno_replace(bm_450.pheno["subtype"] != subtype, "subtype", "Other", inplace=False)
    # LDA fitting
    lda = LinearDiscriminantAnalysis()
    X, y = bm_ova.sklearn_conversion("subtype")
    X = lda.fit_transform(X, y)
    # A RF classifier is trained using the reduced-dimensionality representation of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    print("Accuracy of {} vs All: {}".format(subtype, rfc.score(X_test, y_test)))
    # Plot of the LDA coefficients
    # plot_lda_coefficients(lda.scalings_)
