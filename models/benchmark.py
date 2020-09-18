from methylnet_utils import split_methylation_array_by_pheno
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def prepare_dataset(dataset, output_target):
    train, test, validation = split_methylation_array_by_pheno(dataset, output_target)
    x_train = train["beta"].append(validation["beta"])
    y_train = train["pheno"].append(validation["pheno"]).values.ravel()
    x_test = test["beta"]
    y_test = test["pheno"].values.ravel()
    return x_train, y_train, x_test, y_test


def benchmark_svm(dataset, output_target):
    x_train, y_train, x_test, y_test = prepare_dataset(dataset, output_target)
    params = {
        "C": [0.01, 0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf"]
    }

    svm = SVC()
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=params,
        scoring='accuracy',
        cv=10,
        verbose=0,
        n_jobs=10
    )
    grid_search.fit(x_train, y_train)
    return grid_search.score(x_test, y_test)


def benchmark_rf(dataset, output_target):
    x_train, y_train, x_test, y_test = prepare_dataset(dataset, output_target)
    params = {
        "n_estimators": [500, 1000, 2000],
        "max_features": ["auto", "sqrt", "log2"]
    }

    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=params,
        scoring='accuracy',
        cv=10,
        verbose=0,
        n_jobs=10
    )
    grid_search.fit(x_train, y_train)
    return grid_search.score(x_test, y_test)


def benchmark_knn(dataset, output_target):
    x_train, y_train, x_test, y_test = prepare_dataset(dataset, output_target)
    params = {
        "n_neighbors": [5, 10, 50, 100]
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=params,
        scoring='accuracy',
        cv=10,
        verbose=0,
        n_jobs=10
    )
    grid_search.fit(x_train, y_train)
    return grid_search.score(x_test, y_test)
