from cardio_datasets import reduced_dataset
from model_selection_and_evaluation_functions import create_X_y_from_data_frame
import joblib
import pandas

cardio_model = joblib.load("./models/final/VotingClassifier.joblib")
carido_samples = pandas.read_csv("./data/cardio_100_samples.csv")

X, y = create_X_y_from_data_frame(carido_samples[reduced_dataset.columns], "cardio")


def main():
    pandas.DataFrame(
        cardio_model.predict_proba(X).tolist(),
        columns=["Positive probability", "Negative probability"],
    ).join(
        pandas.Series(cardio_model.predict(X), name="Prediction").map(
            {0: "Positive", 1: "Negative"}
        )
    ).to_csv(
        "./data/cardio_predictions.csv"
    )


if __name__ == "__main__":
    main()
