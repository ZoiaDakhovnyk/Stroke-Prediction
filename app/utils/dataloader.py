import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()
        return self

    def load_data(self):
        #replace
        self.dataset['gender'] = self.dataset['gender'].replace('Other','Female')

        #fill Nan with median
        self.dataset['bmi'] = self.dataset['bmi'].fillna(self.dataset['bmi'].median())

        #drop columns
        self.dataset = self.dataset.drop('id',axis =1)

        #encode labels
        le = LabelEncoder()

        le.fit(self.dataset['gender'])
        self.dataset['gender'] = le.transform(self.dataset['gender'])

        le.fit(self.dataset['ever_married'])
        self.dataset['ever_married'] = le.transform(self.dataset['ever_married'])

        le.fit(self.dataset['work_type'])
        self.dataset['work_type'] = le.transform(self.dataset['work_type'])

        le.fit(self.dataset['Residence_type'])
        self.dataset['Residence_type'] = le.transform(self.dataset['Residence_type'])

        le.fit(self.dataset['smoking_status'])
        self.dataset['smoking_status'] = le.transform(self.dataset['smoking_status'])

        #standartize
        self.dataset['age'] = (self.dataset['age'] - self.dataset['age'].mean()) / self.dataset['age'].std()
        self.dataset['avg_glucose_level'] = (self.dataset['avg_glucose_level'] - self.dataset['avg_glucose_level'].mean()) / self.dataset['avg_glucose_level'].std()
        self.dataset['bmi'] = (self.dataset['bmi'] - self.dataset['bmi'].mean()) / self.dataset['bmi'].std()

        return self.dataset



