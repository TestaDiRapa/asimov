from methylnet_utils import split_methylation_array_by_pheno
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def prepare_dataset(dataset, output_target):
    train, test, validation = split_methylation_array_by_pheno(dataset, output_target, seed=42)
    x_train = train["beta"]
    y_train = train["pheno"].values.ravel()
    x_test = test["beta"]
    y_test = test["pheno"].values.ravel()
    x_val = validation["beta"]
    y_val = validation["pheno"].values.ravel()
    return x_train, y_train, x_test, y_test, x_val, y_val


def benchmark_svm(dataset, output_target):
    x_train, y_train, x_test, y_test, x_val, y_val = prepare_dataset(dataset, output_target)
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    val_acc, test_acc = 0, 0
    for params in ParameterGrid(param_grid):
        svm = SVC(**params)
        svm.fit(x_train, y_train)
        val_acc_tmp = svm.score(x_val, y_val)
        if val_acc_tmp > val_acc:
            val_acc = val_acc_tmp
            test_acc = svm.score(x_test, y_test)
    return val_acc, test_acc


def benchmark_rf(dataset, output_target):
    x_train, y_train, x_test, y_test, x_val, y_val = prepare_dataset(dataset, output_target)
    param_grid = {
        "n_estimators": [2000],
        "max_features": ["auto", "sqrt"]
    }

    val_acc, test_acc = 0, 0
    for params in ParameterGrid(param_grid):
        svm = RandomForestClassifier(**params)
        svm.fit(x_train, y_train)
        val_acc_tmp = svm.score(x_val, y_val)
        if val_acc_tmp > val_acc:
            val_acc = val_acc_tmp
            test_acc = svm.score(x_test, y_test)
    return val_acc, test_acc


def benchmark_knn(dataset, output_target):
    x_train, y_train, x_test, y_test, x_val, y_val = prepare_dataset(dataset, output_target)
    param_grid = {
        "n_neighbors": [5, 10, 50, 100]
    }

    val_acc, test_acc = 0, 0
    for params in ParameterGrid(param_grid):
        svm = KNeighborsClassifier(**params)
        svm.fit(x_train, y_train)
        val_acc_tmp = svm.score(x_val, y_val)
        if val_acc_tmp > val_acc:
            val_acc = val_acc_tmp
            test_acc = svm.score(x_test, y_test)
    return val_acc, test_acc
