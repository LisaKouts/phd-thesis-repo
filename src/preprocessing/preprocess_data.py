import numpy as np
import pandas as pd
import copy
import os

class preprocess_datasets():
    def __init__(self, dataset_name, path = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_name
        if path:
            self.dataset_path = os.path.join(path, self.dataset_name)
        
    def preprocess_dataset(self, df = None):
        '''
        Preprocess and 
        Attibutes
        ---------
        dataset_name: string, available names: 'german.data', 'synthetic.csv', 'BROWARD_CLEAN.csv', 'titanic.csv', 'adult.data'

        Returns
        -------
        1. a DataFrame object with the dataset with normalized numeric columns
        2. a DataFrame object with the dataset with normalized numeric columns and dummy coded nominal variables cause some metrics cannot handle nominal entries (consistency)
        3. metadata in form {
                'dataset_name' : 'German',
                'prot1' :  'Personal_status_and_sex',
                'prot2' : 'Age_in_years',
                'target' : 'Creditworthiness',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : 'Purpose',
                'priv_group_Personal_status_and_sex' : 'male',
                'unpriv_group_Personal_status_and_sex' : 'female',
                'priv_group_Age_in_years' : 0,
                'unpriv_group_Age_in_years' : 1
            }
        4. a binary version of a protected numeric variable cause some measures cannot handle numeric protected features
        '''

        if 'german' in self.dataset_path:
            # retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
            df = pd.read_csv(self.dataset_path, na_values='?', header=None, sep=' ')
            cols = ['Status_of_existing_checking_account','Duration_in_month', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings_accountbonds', 'Present_employment_since', 
                    'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex', 'Other_debtorsguarantors', 'Present_residence_since', 
                    'Property', 'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank', 'Job', 'Number_of_people_being_liable_to_provide_maintenance_for', 'Telephone', 'Foreign_worker', 'Creditworthiness']
            # sorting values for vizualization purposes
            df.columns = cols
            df = df.sort_values(by = 'Creditworthiness', axis=0, kind = 'stable', ignore_index=True)

            # convert to 0 and 1
            df['Creditworthiness'] = np.where(df['Creditworthiness'].values == 1, 1, 0) 
            # recode 'Personal_status_and_sex' based on AIF360's preprocessing
            df['Personal_status_and_sex'] = np.where(df['Personal_status_and_sex'] == 'A92', 'female', 'male')

            age_binary = pd.Series(np.where(df['Age_in_years']>25,int(0),int(1)), index=df.index, name='Age_in_years')

            # normalize numeric variables
            numeric = [False if df[col].dtype == 'object' else True for col in df]
            num=df.loc[:,numeric].values
            scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
            df[df.columns[numeric]] = pd.DataFrame(scaled, columns=df.columns[numeric])

            df_num = copy.deepcopy(df)
            df_num['Personal_status_and_sex'] = np.where(df_num['Personal_status_and_sex'] == 'female', 0.0, 1.0).astype('int64')
            df_num = pd.get_dummies(df_num, drop_first=True, dtype='int64')

            metadata = {
                'dataset_name' : 'German',
                'prot1' :  'Personal_status_and_sex',
                'prot2' : 'Age_in_years',
                'target' : 'Creditworthiness',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : 'Purpose',
                'priv_group_Personal_status_and_sex' : 'male',
                'unpriv_group_Personal_status_and_sex' : 'female',
                'priv_group_Age_in_years' : 0,
                'unpriv_group_Age_in_years' : 1
            }

            return df, df_num, metadata, pd.DataFrame(num,columns=df.columns[numeric]), age_binary
          
        if self.dataset_name == 'BROWARD_CLEAN.csv':
            # retrieved from https://farid.berkeley.edu/downloads/publications/scienceadvances17/

            df = pd.read_csv(self.dataset_path, na_values='?')

            df.drop('id',axis=1,inplace=True)
            df.drop('two_year_recid',axis=1,inplace=True)
            df.drop('compas_decile_score',axis=1,inplace=True)
            df = df.loc[:,df.columns[:-4]] # remove last empty 4 columns

            # rename column for easier processing
            df.rename(columns={"charge_degree (misd/fel)": "charge_degree"}, inplace=True)
            df['charge_id'] = df['charge_id'].astype('object')

            df['race'] = df.race.replace({1: 'Caucasian', 2: 'Black', 3: 'Hispanic', 4: 'Asian', 5: 'Native American', 6: 'Other'})
            df = df.loc[(df['race'] == 'Caucasian') | (df['race'] == 'Black')] # choose only black or white people as did Farid & Dressel

            num_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'priors_count'] # define numeric columns
            df_num = df.loc[:,num_cols]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = scaled

            # create dummy-coded version of the dataset
            df_num2 = copy.deepcopy(df)
            
            df_num2 = pd.get_dummies(df_num2, drop_first=True, dtype='int64')
            df_num2.rename(columns={"race_Caucasian" : "race"},inplace=True) # rename dummy to original variable

            df_num2['compas_guess'] = df_num2.pop('compas_guess') # place target in the last column
            
            # add metadata
            metadata = {
                'dataset_name' : 'BROWARD_CLEAN',
                'prot1' :  'race',
                'prot2' : 'sex',
                'target' : 'compas_guess',
                'pos_label' : 0,
                'neg_label' : 1,
                'R' : 'charge_id',
                'priv_group_race' : 'Caucasian',
                'unpriv_group_race' : 'Black',
                'priv_group_sex' : 1,
                'unpriv_group_sex' : 0, 
            }

            return df, df_num2, metadata, df_num, None

        if self.dataset_name == 'titanic.csv':
            # retrieved from https://biostat.app.vumc.org/wiki/Main/DataSets (titanic3.csv)
            # preprocessing of title https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9
            
            df = pd.read_csv(self.dataset_path, na_values=np.nan)

            df['title'] = df.name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

            normalized_titles = {
                "Capt":       "Officer",
                "Col":        "Officer",
                "Major":      "Officer",
                "Jonkheer":   "Royalty",
                "Don":        "Royalty",
                "Sir" :       "Royalty",
                "Dr":         "Officer",
                "Rev":        "Officer",
                "the Countess":"Royalty",
                "Dona":       "Royalty",
                "Mme":        "Mrs",
                "Mlle":       "Miss",
                "Ms":         "Mrs",
                "Mr" :        "Mr",
                "Mrs" :       "Mrs",
                "Miss" :      "Miss",
                "Master" :    "Master",
                "Lady" :      "Royalty"
            }
            # map the normalized titles to the current titles 
            df.title = df.title.map(normalized_titles)
            # drop PassengerId, Name, Ticket
            df.drop(['name', 'ticket', 'boat', 'home.dest', 'body', 'cabin'], axis=1, inplace=True)

            df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title', 'survived']] # 

            # replace missing values
            df['age'] = np.where(df['age'] == '?',np.median(df['age'][df['age'] != '?'].astype('float64')),df['age']).astype('float64')
            df['fare'] = np.where(df['fare'] == '?',np.median(df['fare'][df['fare'] != '?'].astype('float64')),df['fare']).astype('float64')

            # discretize numeric sensitive feature
            pclass_binary = pd.Series(np.where(df['pclass'] > 2, 1, 0), index=df.index, name='pclass')

            # normalize numeric features
            num_cols = df.select_dtypes(exclude='object').columns
            df_num = df.loc[:,num_cols]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = pd.DataFrame(scaled, columns=df_num.columns)

            # encode nominal features
            df_num2 = copy.deepcopy(df)
            df_num2 = pd.get_dummies(df_num2, drop_first=True, dtype='int64')
            df_num2.rename(columns={"sex_male" : "sex"},inplace=True) # rename dummy to original variable

            metadata = {
                'dataset_name' : 'titanic',
                'prot1' :  'sex',
                'prot2' : 'pclass',
                'target' : 'survived',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : None,
                'priv_group_sex' : 'female',
                'unpriv_group_sex' : 'male',
                'priv_group_pclass' : 1,
                'unpriv_group_pclass' : 0
            }

            return df, df_num2, metadata, df_num, pclass_binary

        if self.dataset_name == 'adult.data':
            # https://archive.ics.uci.edu/ml/datasets/adult
            
            df = pd.read_csv(self.dataset_path, na_values='?', header=None)
            names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                    'hours-per-week', 'native-country', 'income']

            df.columns = names
            num_cols = df.select_dtypes(exclude='object').columns

            # normalize numeric features
            df_num = df.loc[:,list(df[num_cols].columns)]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = pd.DataFrame(scaled, columns=df_num.columns)

            # recode native-country because its granularity is too high and after dummy coding there are too many extra columns
            df['native-country'] = np.where(df['native-country']== ' United-States',0,1)
            def group_race(x):
                if x == " White":
                    return 1.0
                else:
                    return 0.0
            df['race'] = df['race'].apply(lambda x: group_race(x))
            df['sex'] = df['sex'].replace({' Female': 0.0, ' Male': 1.0})
            df.drop(['education'],axis=1,inplace=True)

            # encode nominal attributes such that algorithms can better process them
            df['income'] = np.where(df['income']==" <=50K",0,1)
            df_num2 = pd.get_dummies(df, drop_first=True, dtype='int64')
        
            metadata = {
                'dataset_name' : 'adult',
                'prot1' :  'race',
                'prot2' : 'sex',
                'target' : 'income',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : 'occupation',
                'priv_group_race' : 1,
                'unpriv_group_race' : 0,
                'priv_group_sex' : 1.0,
                'unpriv_group_sex' : 0.0
            }

            return df, df_num2, metadata, df_num, None
        
        if self.dataset_name == 'lawschool.csv':
            
            df = pd.read_csv(self.dataset_path, index_col=0)
            df.drop(['GPA_cat','LSAT_cat'], axis=1, inplace=True)

            # discretize GPA to compute conditional demographic disparity based on https://www.ilrg.com/rankings/law/index/1/desc/GPALow
            GPA_cat = pd.cut(df['GPA'], [0,3.61,3.79,4.3], labels=['low','medium','high'])            
            
            df['resident'] = df['resident'].astype('object')
            df['sex'] = df['sex'].astype('object')
            y = df.pop('admit')

            # normalize ranges
            num_cols = ['LSAT', 'GPA']
            df_num = df.loc[:,num_cols].astype('float')
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = scaled

            # dummy code
            df_num = pd.get_dummies(df, drop_first=True, dtype='int64')
            y = y.replace({'Admitted': 1, 'Not admitted': 0})
            df['admit'] = y.values
            df_num['admit'] = y.values
            df_num['race'] = df_num.pop('race_White')
            df_num['sex'] = df_num.pop('sex_1.0')

            metadata = {
                'dataset_name' : 'lawschool',
                'prot1' :  'race',
                'prot2' : 'sex',
                'target' : 'admit',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : 'GPA',
                'priv_group_race' : 'White',
                'unpriv_group_race' : 'Non-white',
                'priv_group_sex' : 1.0, # male
                'unpriv_group_sex' : 0.0 # female
            }

            return df, df_num, metadata, df.loc[:,num_cols].astype('float'), GPA_cat
        
        if self.dataset_name == 'diabetes.csv':
            # publication: https://5harad.com/papers/fair-ml.pdf
            # download and preprocess: https://github.com/jgaeb/measure-mismeasure/blob/main/diabetes.R

            df = pd.read_csv(self.dataset_path, index_col=0)
            bmi_binary = pd.Series(np.where(df['bmi'] > 30, 1, 0), index=df.index, name='bmi')
            df.drop('weights',axis=1,inplace=True)
            y = df.pop('diabetes')
            df_num = df.select_dtypes(['int','float'])
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[df.select_dtypes(['int','float']).columns] = scaled
            df['diabetes'] = y.astype(int).values

            metadata = {
                'dataset_name' : 'diabetes',
                'prot1' :  'race',
                'prot2' : None,
                'target' : 'diabetes',
                'pos_label' : 0,
                'neg_label' : 1,
                'R' : 'bmi', # for now
                'priv_group_race' : 'White',
                'unpriv_group_race' : 'Black', # not sure, check paper
            }

            df_num2 = copy.deepcopy(df)
            df_num2 = pd.get_dummies(df_num2, drop_first=True, dtype='int32')
            df_num2['race'] = df_num2.pop('race_Black')
            df_num2['diabetes'] = df_num2.pop('diabetes')

            return df, df_num2, metadata, df_num, bmi_binary
        
        if 'bank' in self.dataset_name:
            df = pd.read_csv(self.dataset_path, sep=';')
            age_binary = pd.Series(np.where(df['age']>35,int(0),int(1)), index=df.index, name='age')

            y = df.pop('y')
            y = y.replace({'yes': 1, 'no': 0})

            df_num = df.select_dtypes(['int','float'])
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[df.select_dtypes(['int','float']).columns] = scaled
            df['deposit'] = y.astype(int).values
            df['marital'] = np.where(df['marital'] == 'married', 1, 0)

            df_num2 = copy.deepcopy(df)
            df_num2 = pd.get_dummies(df_num2, drop_first=True, dtype='int32')
            df_num2['deposit'] = df_num2.pop('deposit')

            metadata = {
                'dataset_name' : 'bank',
                'prot1' :  'marital',
                'prot2' : 'age',
                'target' : 'deposit',
                'pos_label' : 0,
                'neg_label' : 1,
                'R' : 'default', # for now
                'priv_group_marital' : 0,
                'unpriv_group_marital' : 1 , # not sure, check paper
                'priv_group_age' : 0,
                'unpriv_group_age' : 1
            }

            return df, df_num2, metadata, df_num, age_binary
        
        if self.dataset_name == 'acsincome.pkl':
            # Predict whether US working adults’ yearly income is above $50,000
            # file:///C:/Users/lucp11124/Downloads/3540261.3540757_supp.pdf

            df = pd.read_pickle(self.dataset_path)
            df['y'] = df['y'].astype(int)

            # older than 16 (folktables documentation)
            df = df[df['AGEP'] > 16]

            # recode place of birth to US citizen or not to reduce dimensionality
            df['POBP'] = pd.Series(np.where(df['POBP'] <100, 'UScitizen', 'Foreigner'), index=df.index, name='POBP')

            # keep white and black people
            df = df[(df['RAC1P'] == 1) | (df['RAC1P'] == 2)]
            df['RAC1P'] = np.where(df['RAC1P'] == 1, 'White', 'Black')

            # recode sex
            df['SEX'] = np.where(df['SEX'] == 1, 'Male', 'Female')

            # WKHP (Usual hours worked per week past 12 months): Must be greater than 0 (folktables documentation)
            df = df[df['WKHP'] > 0]

            # cuts based occupation type from here https://data.census.gov/mdat/#/search?ds=ACSPUMS1Y2023&rv=OCCP&wt=PWGTP
            df['OCCP'] = pd.cut(df['OCCP'],
                [0,500,800,1000,1300,1600,2000,2100,2200,2600,3000,3600,3700,4000,4200,4300,4700,5000,6000,6200,6800,7000,7700,9000,9800,9900,10000],
                labels=['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT', 'CLN', 'PRS', 'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'RPR', 'PRD', 'TRN', 'MIL', 'UNEMP']).astype('object')

            df['COW'] = df['COW'].astype('object')
            df['SCHL'] = df['SCHL'].astype('object')
            df['MAR'] = df['MAR'].astype('object')
            df['RELP'] = df['RELP'].astype('object')

            numeric = [False if df[col].dtype == 'object' else True for col in df]
            num=df.loc[:,numeric].values
            scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
            df[df.columns[numeric]] = scaled

            df_num = copy.deepcopy(df)
            df_num = pd.get_dummies(df_num, drop_first=True, dtype='int64')
            df_num['RAC1P'] = df_num.pop('RAC1P_White')
            df_num['SEX'] = df_num.pop('SEX_Male')
            df_num['y'] = df_num.pop('y')

            metadata = {'dataset_name' : 'acsincome',
                        'prot1' :  'RAC1P',
                        'prot2' : 'SEX',
                        'target' : 'y',
                        'pos_label' : 1, # higher than 50k
                        'neg_label' : 0, # lower than 50k
                        'R' : 'OCCP',
                        'priv_group_RAC1P' : 'White',
                        'unpriv_group_RAC1P' : 'Black',
                        'priv_group_SEX' : 'Male',
                        'unpriv_group_SEX' : 'Female'
                        }
            
            return df, df_num, metadata, pd.DataFrame(num,columns=df.columns[numeric]), None
        
        if self.dataset_name == 'acspubliccoverage.pkl':
            df = pd.read_pickle(self.dataset_path)
            df['y'] = df['y'].astype(int)

            # older than 16 (folktables documentation)
            df = df[df['AGEP'] < 65]
            df = df[df['PINCP'] < 30000]

            # recode sex
            df['SEX'] = np.where(df['SEX'] == 1, 'Male', 'Female')

            # recode disability
            df['DIS'] = np.where(df['DIS'] == 1, 'With disability', 'Without disability')

            # keep white and black people
            df = df[(df['RAC1P'] == 1) | (df['RAC1P'] == 2)]
            df['RAC1P'] = np.where(df['RAC1P'] == 1, 'White', 'Black')

            df['SCHL'] = df['SCHL'].astype('object')
            df['MAR'] = df['MAR'].astype('object')
            df['ESP'] = df['ESP'].astype('object')
            df['CIT'] = df['CIT'].astype('object')
            df['MIG'] = df['MIG'].astype('object')
            df['MIL'] = df['MIL'].astype('object')
            df['ANC'] = df['ANC'].astype('object')
            df['ESR'] = df['ESR'].astype('object')
            df['ST'] = df['ST'].astype('object')
            df['FER'] = df['FER'].astype('object')

            numeric = [False if df[col].dtype == 'object' else True for col in df]
            num=df.loc[:,numeric].values
            scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
            df[df.columns[numeric]] = scaled

            df_num = copy.deepcopy(df)
            df_num = pd.get_dummies(df_num, drop_first=True, dtype='int64')
            df_num['RAC1P'] = df_num.pop('RAC1P_White')
            df_num['SEX'] = df_num.pop('SEX_Male')
            df_num['y'] = df_num.pop('y')

            metadata = {'dataset_name' : 'acspubliccoverage',
                        'prot1' :  'RACE',
                        'prot2' : 'SEX',
                        'target' : 'y',
                        'pos_label' : 1, # with public health coverage
                        'neg_label' : 0, # without public health coverage
                        'R' : 'OCCP',
                        'priv_group_RACE' : 'White',
                        'unpriv_group_RACE' : 'Black',
                        'priv_group_SEX' : 'Male',
                        'unpriv_group_SEX' : 'Female'
                        }
            
            return df, df_num, metadata, pd.DataFrame(num,columns=df.columns[numeric]), None
        
        if self.dataset_name == 'acsemployment.pkl':
            # predict whether an adult is employed
            # Target: ESR (Employment status recode): an individual’s label is 1 if ESR == 1, and 0 otherwise.

            df = pd.read_pickle(self.dataset_path)

            # older than 16 (folktables documentation)
            df = df[df['AGEP'] > 16]
            df.drop('NATIVITY',axis=1,inplace=True) # 100% correlation with CIT, therefore removed
            
            # dropping disability because 'DIS' conveys the similar information
            df.drop('DEAR',axis=1,inplace=True)
            df.drop('DEYE',axis=1,inplace=True)
            df.drop('DREM',axis=1,inplace=True)

            # keep white and black people
            df = df[(df['RAC1P'] == 1) | (df['RAC1P'] == 2)]
            df['RAC1P'] = np.where(df['RAC1P'] == 1, 'White', 'Black')
            df['RACE'] = df.pop('RAC1P')

            # recode sex
            df['SEX'] = np.where(df['SEX'] == 1, 'Male', 'Female')

            # Recode target from boolean to numeric
            df['y'] = df.pop('y').astype(int)

            # specify nominal features
            df['DIS'] = df['DIS'].astype('object')
            df['MAR'] = df['MAR'].astype('object')
            df['ESP'] = df['ESP'].astype('object')
            df['MIL'] = df['MIL'].astype('object')
            df['CIT'] = df['CIT'].astype('object')
            df['MIG'] = df['MIG'].astype('object')
            df['ANC'] = df['ANC'].astype('object')
            df['RELP'] = df['RELP'].astype('object')

            numeric = [False if df[col].dtype == 'object' else True for col in df]
            num=df.loc[:,numeric].values
            scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
            df[df.columns[numeric]] = scaled

            df_num = copy.deepcopy(df)
            df_num = pd.get_dummies(df_num, drop_first=True, dtype='int64')
            df_num['RACE'] = df_num.pop('RACE_White')
            df_num['SEX'] = df_num.pop('SEX_Male')
            df_num['y'] = df_num.pop('y')

            metadata = {'dataset_name' : 'acsemployment',
                        'prot1' :  'RACE',
                        'prot2' : 'SEX',
                        'target' : 'y',
                        'pos_label' : 1, # with public health coverage
                        'neg_label' : 0, # without public health coverage
                        'R' : 'DIS',
                        'priv_group_RACE' : 'White',
                        'unpriv_group_RACE' : 'Black',
                        'priv_group_SEX' : 'Male',
                        'unpriv_group_SEX' : 'Female'
                        }
            
            return df, df_num, metadata, pd.DataFrame(num,columns=df.columns[numeric]), None

# import sys
# if __name__=="__main__":
#   args = Writearray(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]).sim_array(sys.argv[5], sys.argv[6], sys.argv[7])
#   print("In mymodule:",args)

