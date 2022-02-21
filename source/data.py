import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
       
def data_cleaning(df, file):		
       if file == 'cervical-cancer':
          df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'Hinselmann', 'Schiller', 'Citology'],axis=1)
       elif file == 'credit-score':
          df = df.drop(['Unnamed: 0'],axis=1)
       elif file == 'adult-income': 
          pass 
       df = self.fill_na(df,file)
       return df
       
def fill_na(df, file):		
       if file == 'cervical-cancer':
          df = df.replace('?',np.NaN)
          df = df.dropna(how='any')
       elif file == 'credit-score':
          df[['Checking account','Saving accounts']] = df[['Checking account','Saving accounts']].fillna('')
       elif file == 'adult-income': 
          pass
       return df
       
def transform_label(df, name):
       LE = LabelEncoder()		
       if name == 'cervical-cancer':
          df['Biopsy'] = LE.fit_transform(df['Biopsy'])
       elif name == 'credit-score':
          df['Risk'] = LE.fit_transform(df['Risk'])
       elif name == 'adult-income': 
          df['income'] = LE.fit_transform(df['income'])
       return df
    

def balance_dataset(df,name):
    
    ros = SMOTE()
    rus = RandomUnderSampler()
    
    if name == 'cervical-cancer':
        target = df["Biopsy"]
        X=df.drop(['Biopsy'],axis=1)
        df, target = ros.fit_resample(X,target)
        df = pd.concat([df,target],axis=1)
        for col in df.columns:
            df[col] = df[col].astype(str).astype(float)
            
    elif name == 'credit-score':
        target = df["Risk"]
        X=df.drop(['Risk'],axis=1)
        df, target = rus.fit_resample(X,target)
        df = pd.concat([df,target],axis=1)
        
    elif name == 'adult-income':
        target = df["income"]
        X=df.drop(['income'],axis=1)
        df, target = rus.fit_resample(X,target)
        df = pd.concat([df,target],axis=1)
        
    return df

