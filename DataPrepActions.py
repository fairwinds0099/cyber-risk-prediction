import pandas as pd

class DataPrepActions():

    def fetchAndCleanDataframe(self):
        """
            gets raw dataset from local
            drops unused columns
            convert the DataFrame into two variables
            X: data columns (OCEAN)
            y: label column
        """

        df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/results_with_scenarios.csv')
        df.insider_label.fillna(0, inplace=True) # replaces null fields with 0
        df = df.drop(columns=['employee_name', 'scenario', 'role'])
        df = df.rename(columns={'insider_label':'label'})
        #df['label'] = df['insider_label'].astype('int64')
        #df.drop(columns='insider_label', inplace=True)
        df.set_index('user_id', inplace=True)
        X = df.iloc[:, :5].values #fetch all records first 5 columns
        y = df.label.values
        print(df.head())
        return X, y
