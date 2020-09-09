from dataset import identify_breast_cancer_subtype
import os

# Experiment to classify the breast cancer subtypes in TCGA

statuses = {
    "LuminalA": 0,
    "LuminalB": 0,
    "TNBS": 0,
    "HER2+": 0,
    "Unclear": 0
}
for dir_, _, files in os.walk("../data/breast_clinical"):
    for file in files:
        if file[-3:] == "xml":
            path = os.path.join(dir_, file)
            statuses[identify_breast_cancer_subtype(path)] += 1
print(statuses)
