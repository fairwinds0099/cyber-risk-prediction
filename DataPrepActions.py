import pandas as pd

class DataPrepActions:

    def fetchAndCleanDataframe(self):
        """
            gets raw dataset from local
            drops unused columns
            convert the DataFrame into two variables
            X: data columns (OCEAN)
            y: label column
        """

        global df
        df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/results_with_scenarios.csv')
        df.insider_label.fillna(0, inplace=True)
        df = df.drop(columns=['employee_name', 'user_id'])
        df['label'] = df['insider_label'].astype('int64')
        df.drop(columns='insider_label', inplace=True)
        df.drop(columns='scenario', inplace=True)
        df.drop(columns='role', inplace=True)
        print(df.head())
        X = df.iloc[:, :5].values
        y = df.label.values
        return X, y
