from omic_array import OmicArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# OBJECTIVE: feature extraction using LDA

# Loading the methylation 450k values for breast cancer
bm_450 = OmicArray("../../data/breast/methylation/breast_450k_final.pkl")
# One-vs-All analysis using only the Luminal A class
bm_450.pheno_replace(bm_450.pheno["subtype"] != "Luminal A", "subtype", "Other")

lda = LinearDiscriminantAnalysis()
lda.fit(*bm_450.sklearn_conversion("subtype"))
print(lda.scalings_)