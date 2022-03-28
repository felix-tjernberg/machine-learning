import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def create_X_y_from_data_frame(data_frame: pandas.DataFrame, target_column: str):
    """Takes a pandas DataFrame and string of target column and returns X, y"""
    return data_frame.drop(columns=[target_column]), data_frame[target_column]


def grid_search_hyper_parameters(
    X, y, model, parameter_grid: dict, scoring_method: str
):
    model = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        return_train_score=False,
        scoring=scoring_method,
        cv=5,
    )
    model.fit(X, y)
    return model, pandas.DataFrame(model.cv_results_)


def select_search_parameters_and_scores(
    results_data_frame: pandas.DataFrame, parameter_grid: dict
):
    columns = ["param_" + key for key in list(parameter_grid.keys())] + [
        "mean_test_score",
        "std_test_score",
    ]
    return (
        results_data_frame[columns]
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )


def search_hyper_parameters(X, y, model, parameter_grid: dict, scoring_method: str):
    model, scores = grid_search_hyper_parameters(
        X, y, model, parameter_grid, scoring_method
    )
    return model, select_search_parameters_and_scores(scores, parameter_grid), scores


def show_evaluation_metrics(model, X_test, y_test, display_labels=["True", "False"]):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=display_labels
    ).plot()


def search_score_and_evalute_parameters(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    model_parameter_grid: dict,
    scoring_method: str,
):
    (
        fitted_model,
        model_parameters_and_score,
        model_score_raw,
    ) = search_hyper_parameters(
        X_train, y_train, model, model_parameter_grid, scoring_method
    )
    show_evaluation_metrics(fitted_model, X_test, y_test)
    return fitted_model, model_parameters_and_score, model_score_raw
