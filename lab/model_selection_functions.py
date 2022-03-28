import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def create_X_y_from_data_frame(data_frame: pandas.DataFrame, target_column: str):
    """Return a pandas.DataFrame's without target column and target column"""
    return data_frame.drop(columns=[target_column]), data_frame[target_column]


def grid_search_hyper_parameters(
    X, y, model, parameter_grid: dict, scoring_method: str
):
    """Retruns a fitted model and a pandas.DataFrame of GridSearchCV.cv_results"""
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
    """Returns a selection of parameters and scores from a GridSearchCV.cv_results_ data frame using the keys from a parameter grid"""
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
    """Returns a fitted model, a selection of parameters and scores of GridSearchCV.cv_results_, raw GridSearchCV.cv_results_"""
    model, scores = grid_search_hyper_parameters(
        X, y, model, parameter_grid, scoring_method
    )
    return model, select_search_parameters_and_scores(scores, parameter_grid), scores


def show_evaluation_metrics(model, X_test, y_test, display_labels=["True", "False"]):
    """Prints classification report and plots confusion matrix"""
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
    """Searches, scores and prints evaluation of a model and it's parameters then returns the model"""
    (
        fitted_model,
        model_parameters_and_score,
        model_score_raw,
    ) = search_hyper_parameters(
        X_train, y_train, model, model_parameter_grid, scoring_method
    )
    show_evaluation_metrics(fitted_model, X_test, y_test)
    return fitted_model, model_parameters_and_score, model_score_raw
