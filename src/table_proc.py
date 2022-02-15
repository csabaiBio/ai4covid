from tkinter.filedialog import test
import pandas as pd

class TableProcessor:
    def __init__(self, config) -> None:
        self.train_data = pd.read_excel(config.train_table_path)
        self.train_data.drop(['Prognosis', 'Death'], axis=1, inplace=True)
        hospitals = pd.get_dummies(self.train_data['Hospital'], prefix_sep='_', prefix='Hospital')
        self.train_data = pd.concat([self.train_data, hospitals], axis=1)
        self.train_data.drop('Hospital', axis=1, inplace=True)
        self.train_data.drop(["Row_number", "ImageFile"], axis=1, inplace=True)
        self.train_data = self.train_data.astype(float)

        self.train_std = self.train_data.std()
        self.train_mean = self.train_data.mean()

    def line_impute_population_average(self, test_data_row):
        """Impute missing values in a row of test data based on average
        value of the 10 closes patient in age.

        Args:
        test_data_row: (DataFrame) A single row of the test dataset
        Returns:
        test_data_imputed (DataFrame): Test dataset with NaN values imputed.
        """
        test_data_row = pd.concat([test_data_row.reset_index().drop('index', axis=1), 
                                        pd.DataFrame({'Hospital_A': [0], 'Hospital_B': [0], 'Hospital_C': [0], 'Hospital_D': [0], 'Hospital_E': [0], 'Hospital_F': [0]})], axis=1)

        test_data_row['Hospital_{}'.format(test_data_row['Hospital'].values[0])] = 1
        test_data_row.drop('Hospital', axis=1, inplace=True)
    
        image_file = test_data_row.pop("ImageFile").values[0]
        test_data_row.drop(["Row_number"], axis=1, inplace=True)

        test_data_row = test_data_row.astype(float)

        for _, col_name in enumerate(test_data_row.columns.values):
            if test_data_row[col_name].isnull().values.any():
                test_data_row[col_name] = self.train_data[col_name].mean()
                
        test_data_row = (test_data_row - self.train_mean)/self.train_std
        
        return image_file, test_data_row.to_numpy().flatten()