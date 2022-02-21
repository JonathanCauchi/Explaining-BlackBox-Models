import dice_ml
from dice_ml.utils import helpers  
import pandas as pd

def get_data(file):
    try:		
       if file == 'cervical-cancer':
          df = pd.read_csv('risk_factors_cervical_cancer.csv')
       elif file == 'credit-score':
          df = pd.read_csv('german_credit_data.csv')	
       elif file == 'adult-income': 
          df = helpers.load_adult_income_dataset()
       return df
    except NameError:
       print("File {} is not found".format(str(file)))
    except:
       print("Something went wrong")
