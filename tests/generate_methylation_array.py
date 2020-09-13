'''
from methylnet_utils import generate_subtype_methylation_array

generate_subtype_methylation_array("../data/breast_clinical",
                                   "../data/breast_methylation_450_pd.pkl",
                                   "../data/breast_methylation_450_ma.pkl")
'''
from methylnet_utils import split_methylation_array_by_pheno
split_methylation_array_by_pheno("../data/final_preprocessed/methyl_array.pkl",
                                 "subtype",
                                 "../data/train_val_test_sets")
