import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skfuzzy.cluster import cmeans
from sklearn.model_selection import train_test_split
from scipy.stats.contingency import association
from itertools import product
import skfuzzy

def train_classifier_predict(df_num, metadata, seed, predict = False, test = False):
    '''
    Function that splits the dataset, train a random forest classifier and returns the trained model and the test set indices.

    Attributes:
    -----------
    df_num: pandas DataFrame, all variables need to be numeric, nominal ones need to be dummy coded, such that it can be processed by the RandomForestClassifier
    
    metadata: dictionary, an example is {
                'dataset_name' : 'German',
                'prot1' :  'Personal_status_and_sex',
                'prot2' : 'Age_in_years',
                'target' : 'Creditworthiness',
                'pos_label' : 1}

    seed: integer, it is used for the random_state parameter of the train_test_split function to make sure that the same train-test sets are created for the crossvalidation 

    predict: boolean, if True the classifier also makes predictions and returns them, if False it returns the trained classifier and the test set. Predictions are optional to save time and computing power.

    Returns:
    --------
    It returns three objects. The first is the test set from which the index is then retrieved and used subsequently. The second is the test set labels. The third is the train classifier which will be later used to make predictions.
    '''
    X = df_num[[col for col in df_num.columns if metadata['target'] not in col]]
    y = df_num[metadata['target']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    clf = RandomForestClassifier(random_state=42).fit(X_train, y_train.astype('int64'))
    if predict:
        y_pred = clf.predict(X_test)
        return y_test, y_pred, X_test
    else:
        return X_test, y_test, clf

def discretize_fcmeans(df, var):
    '''
    Discretizes numeric variable using fuzzy c-means, Source: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

    Attributes
    ----------
    df: pandas dataframe with normalized numeric variables
    var: sting, name of variable in dataframe to discretize

    Returns
    ----------
    Vector with discretized new variable
    '''
    xpts = df[var].values.reshape(-1, 1) # select variable
    xpts = np.vstack((xpts.flatten(), xpts.flatten())).astype(np.float64) # dulicate variable

    fpc_best = 0
    cntr_best = 0
    for n_centers in range(2,10):
        cntr, _, _, _, _, _, fpc = cmeans(xpts, n_centers, 2, error=0.005, maxiter=1000, init=None) # https://scikit-fuzzy.github.io/scikit-fuzzy/api/index.html, https://scikit-fuzzy.github.io/scikit-fuzzy/auto_examples/plot_cmeans.html
        if fpc_best < fpc:
            fpc_best, cntr_best = fpc, cntr

    diff = df[var].values.reshape((-1,1)) - cntr_best[:,0].reshape((1,-1)) 
    distance = ((diff)**2)**0.5
    return np.argmin(distance, axis=1).astype('object')


def CDD_difference(df, metadata, prot):
    '''
    df : pandas dataframe, data to compute CDD on
    metadata: dictionary which should look like that, metadata = {
                'dataset_name' : 'BROWARD_CLEAN',
                'prot1' : 'race',
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
    prot: string, name of protected feature       
    '''
    sizes = pd.DataFrame(df.groupby([prot, metadata['target'], metadata['R']]).size(), columns = ['Number '+metadata['target']])
    sizes['Rate '+metadata['target']] = sizes / sizes.groupby([metadata['target'],metadata['R']]).transform('sum')
    pivot = sizes.reset_index().pivot(columns = [metadata['target'], prot], index = metadata['R'], values = ['Rate '+metadata['target'],])

    weights = pd.DataFrame(np.unique(df[metadata['R']], return_counts=True)).T
    weights.columns = [metadata['R'], 'Weights']
    weights.set_index(metadata['R'], inplace = True)

    cdd = []
    for pair in pivot.columns:
        group_label = pd.DataFrame(pivot[pair[0]][pair[1]][pair[2]])
        group_label['Weights'] = weights['Weights']
        group_label.fillna(0, inplace=True)
        cdd_agg = round(sum(group_label[pair[2]] * group_label['Weights']) / len(df[metadata['R']]),2)
        cdd.append([pair[1],pair[2],cdd_agg])

    cdd_groups = pd.DataFrame(cdd, columns=[metadata['target'], prot, 'cdd'])
    unpriv_fav = cdd_groups[(cdd_groups[metadata['target']] == metadata['pos_label']) & (cdd_groups[prot] == metadata['unpriv_group_'+prot])]['cdd'].values[0]
    unpriv_unfav = cdd_groups[(cdd_groups[metadata['target']] == metadata['neg_label']) & (cdd_groups[prot] == metadata['unpriv_group_'+prot])]['cdd'].values[0]

    return unpriv_fav - unpriv_unfav #, min(unpriv_fav/unpriv_unfav, unpriv_unfav/unpriv_fav)

def discretize_fcmeans(df, var):
    '''
    Discretizes numeric variable using fuzzy c-means, Source: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

    Attributes
    ----------
    df: pandas dataframe with normalized numeric variables
    var: sting, name of variable in dataframe to discretize

    Returns
    ----------
    Vector with discretized new variable
    '''
    xpts = df[var].values.reshape(-1, 1) # select variable
    xpts = np.vstack((xpts.flatten(), xpts.flatten())).astype(np.float64) # dulicate variable

    fpc_best = 0
    cntr_best = 0
    for n_centers in range(2,10):
        cntr, _, _, _, _, _, fpc = skfuzzy.cluster.cmeans(xpts, n_centers, 2, error=0.005, maxiter=1000, init=None, seed = 42) # https://scikit-fuzzy.github.io/scikit-fuzzy/api/index.html, https://scikit-fuzzy.github.io/scikit-fuzzy/auto_examples/plot_cmeans.html
        if fpc_best < fpc:
            fpc_best, cntr_best = fpc, cntr

    diff = df[var].values.reshape((-1,1)) - cntr_best[:,0].reshape((1,-1)) 
    distance = ((diff)**2)**0.5
    return (np.argmin(distance, axis=1) + 1).astype('object')

def cramers_V(df, features):
    '''
    Cramer's V association statistic based on p. 112, paragraph 3.33 of Agresti, Alan. Categorical data analysis. Vol. 792. John Wiley & Sons, 2012.

    Attributes
    ----------
    df: pandas dataframe with nominal features
    features: features to compute association on

    Returns
    ----------
    square correlation matrix
    '''
    corr_matrix = pd.DataFrame(columns=features, index=features)

    # populate square matrix with Cramer's V correlation coefficient
    for x,y in list(product(features, features)):
        ct = pd.crosstab(df[x], df[y]).values
        cv = association(ct, method="cramer")
        corr_matrix.loc[x,y] = cv
        corr_matrix.loc[y,x] = cv
    
    return corr_matrix

def onehot_aggregator(values, nominal_features, onehot_features, original_columns, metadata):
    '''
    nominal: boolean array, if True it is a nominal variable, if False it is a numeric variable
    '''
    
    df = pd.DataFrame(data=values, columns=onehot_features)

    # collect the dummy variables
    for f_norm in nominal_features:
        dummies = []
        for f_dummy in onehot_features:
            if f_norm in f_dummy:
                dummies.append(f_dummy)
        
        # take the sum of dummy variables of a single feature, remove the dummies and add a new column with the sum
        dummy_sum = df[dummies].sum(axis=1)
        df = df.drop(dummies, axis=1)
        df[f_norm] = dummy_sum
    
    # use the original order of variables
    df = df[original_columns]
    return df