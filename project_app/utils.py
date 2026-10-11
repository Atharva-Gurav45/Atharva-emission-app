import pickle
import json
import pandas as pd
import config


class EmissionClassifier():

    def __init__(self, AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, Year):
        self.AT = AT
        self.AP = AP
        self.AH = AH
        self.AFDP = AFDP
        self.GTEP = GTEP
        self.TIT = TIT
        self.TAT = TAT
        self.TEY = TEY
        self.CDP = CDP
        self.Year = Year

    def load_model(self):
        # model + scaler (pickle)
        with open(config.Model_file, 'rb') as f:
            art = pickle.load(f)

        self.model = art["model"]
        self.scaler = art["scaler"]
        self.columns = art["columns"]

        # feature names (json)
        with open(config.JSON_file, 'r') as f:
            self.project_data = json.load(f)

    def get_predicted_class(self):

        self.load_model()

        # build input in the same column order as training
        features = self.project_data["features"]
        if features != self.columns:
            raise ValueError("JSON features and pickle columns do not match. "
                             "Run the training script again.")

        data = {}
        for col in features:
            try:
                data[col] = float(getattr(self, col))
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' must be a number, got {getattr(self, col)!r}")

        new = pd.DataFrame([data], columns=features)

        # same steps as training: scale everything, then keep Year unscaled
        new_s = pd.DataFrame(self.scaler.transform(new), columns=features)
        new_s["Year"] = new["Year"]

        predicted_class = self.model.predict(new_s)[0]

        print(f'Predicted emission class is: {predicted_class}')

        return predicted_class


if __name__ == "__main__":
    turbine = EmissionClassifier(
        AT=15.0, AP=1010.0, AH=70.0, AFDP=3.5, GTEP=25.0,
        TIT=1080.0, TAT=545.0, TEY=135.0, CDP=12.0, Year=2013,
    )
    predicted = turbine.get_predicted_class()
    print("Predicted Emission Class:", predicted)
