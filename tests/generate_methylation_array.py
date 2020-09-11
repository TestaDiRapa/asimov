from methylnet_utils import generate_subtype_methylation_array

generate_subtype_methylation_array("../data/breast_clinical",
                                   "../data/breast_methylation_450_pd.pkl",
                                   "../data/breast_methylation_450_ma.pkl")
