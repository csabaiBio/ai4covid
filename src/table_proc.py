import numpy as np
import pandas as pd


class TableProcessor:
    def __init__(self, config) -> None:
        self.train_data = pd.read_excel(config.train_table_path)
        self.train_data.drop(["Prognosis", "Death"], axis=1, inplace=True)
        hospitals = pd.get_dummies(
            self.train_data["Hospital"], prefix_sep="_", prefix="Hospital"
        )
        self.train_data = pd.concat([self.train_data, hospitals], axis=1)
        self.train_data.drop(
            ["Hospital", "Row_number", "ImageFile"], axis=1, inplace=True
        )
        self.continuous_features = [
            "Age",
            "Temp_C",
            "DaysFever",
            "WBC",
            "RBC",
            "CRP",
            "Fibrinogen",
            "Glucose",
            "PCT",
            "LDH",
            "INR",
            "D_dimer",
            "Ox_percentage",
            "PaO2",
            "SaO2",
            "PaCO2",
            "pH",
        ]
        self.discrete_features = [
            "Sex",
            "PositivityAtAdmission",
            "Cough",
            "DifficultyInBreathing",
            "CardiovascularDisease",
            "IschemicHeartDisease",
            "AtrialFibrillation",
            "HeartFailure",
            "Ictus",
            "HighBloodPressure",
            "Diabetes",
            "Dementia",
            "BPCO",
            "Cancer",
            "ChronicKidneyDisease",
            "RespiratoryFailure",
            "Obesity",
            "Position",
        ]

    def _preprocess_test_data_row(self, test_data_row):
        test_data_row = pd.concat(
            [
                test_data_row.reset_index().drop("index", axis=1),
                pd.DataFrame(
                    {
                        "Hospital_A": [0],
                        "Hospital_B": [0],
                        "Hospital_C": [0],
                        "Hospital_D": [0],
                        "Hospital_E": [0],
                        "Hospital_F": [0],
                    }
                ),
            ],
            axis=1,
        )

        test_data_row["Hospital_{}".format(test_data_row["Hospital"].values[0])] = 1
        test_data_row.drop(["Hospital", "Row_number"], axis=1, inplace=True)
        image_file = test_data_row.pop("ImageFile").values[0]

        return image_file, test_data_row

    def line_impute_population_average(self, test_data_row):
        """Impute missing values in a row of test data based on mean, meadian
        value of the corresponding column in the training data.

         Args:
         test_data_row: (DataFrame) A single row of the test dataset
         Returns:
         test_data: (DataFrame): A single row of the test dataset with NaN values imputed.
        """

        image_file, test_data_row = self._preprocess_test_data_row(test_data_row)

        for _, col_name in enumerate(test_data_row.columns.values):
            if test_data_row[col_name].isnull().values.any():
                if col_name in self.continuous_features:
                    test_data_row[col_name] = self.train_data[col_name].dropna().mean()
                elif col_name in self.discrete_features:
                    test_data_row[col_name] = (
                        self.train_data[col_name].dropna().median()
                    )

        test_data_row[self.continuous_features] = (
            test_data_row[self.continuous_features]
            - self.train_data[self.continuous_features].mean()
        ) / self.train_data[self.continuous_features].std()

        return image_file, test_data_row.to_numpy().flatten()

    def line_impute_population_sampling(self, test_data_row):
        """Impute missing values in a row of test data based on random
        sampling from the available values in the corresponding column in the training data.

        Args:
        test_data_row: (DataFrame) A single row of the test dataset
        Returns:
        test_data: (DataFrame): A single row of the test dataset with NaN values imputed.
        """
        image_file, test_data_row = self._preprocess_test_data_row(test_data_row)

        for _, col_name in enumerate(test_data_row.columns.values):
            if test_data_row[col_name].isnull().values.any():
                if (
                    col_name in self.continuous_features
                    or col_name in self.discrete_features
                ):

                    test_data_row[col_name] = np.random.choice(
                        self.train_data[col_name].dropna()
                    )

        test_data_row[self.continuous_features] = (
            test_data_row[self.continuous_features]
            - self.train_data[self.continuous_features].mean()
        ) / self.train_data[self.continuous_features].std()

        return image_file, test_data_row.to_numpy().flatten()
