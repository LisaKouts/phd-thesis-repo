import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# copied from https://github.com/rcrupiISP/BiasOnDemand
def create_synth(dim=15000, 
                 l_y=4, 
                 l_m_y=0, 
                 thr_supp=1, 
                 l_h_r=1.5,  
                 l_h_q=1, 
                 l_m=1, 
                 p_u=1, 
                 l_r=False, 
                 l_o=False, 
                 l_y_b=0, 
                 l_q=2, 
                 sy=5, 
                 l_r_q=0, 
                 l_m_y_non_linear=False, 
                 seed = 42,
                 dataset_name = None):
    """Generate a synthetic dataset.

    Parameters
    ----------
    dim : int
        Dimension of the dataset
    l_y : float, optional
        Lambda coefficient for historical bias on the target y (y is decision)
    l_m_y : float, optional
        Lambda coefficient for measurement bias on the target y, range 1-9 as well, closer to 9 
    thr_supp: float, optional
        Threshold correlation for discarding features too much correlated with s
    l_h_r: float, optional
        Lambda coefficient for historical bias on R, is a natural number, range 1-10
    l_h_q: float, optional
        Lambda coefficient for historical bias on Q, beta^{Q}_{h} in math main paper
    l_m: float, optional
        Lambda coefficient for measurement bias. If l_m!=0 P substitute R., is a natural number, range 1-10
    p_u: float, optional
        Percentage of undersampling instance with A=1
    l_r: bool, optional
        Boolean for inducing representation bias, that is undersampling conditioning on a variable, e.g. X2
    l_o: bool, optional
        Boolean variable for excluding an important variable, e.g. X2
    l_y_b: float, optional
        Lambda coefficient for interaction proxy bias
    l_q: float, optional
        Lambda coefficient for importance of Q for Y, Q is an additional variable that may or may not be relevant to estimate Y, alpha_{Q} in math main and supplementary paper
    sy: float, optional
        Standard deviation of the noise of Y
    l_r_q: float, optional
        Lambda coefficient that quantifies the influence from R to Q
    
    Returns
    -------
    list
        a list datasets train and test for: complete dataset, individual and suppression.
    """
    np.random.seed(42)
    # # # dataset
    # Random var
    N1 = np.random.gamma(2, 3, dim) # 2 is k_R, 3 is theta_R denoting the shape and scale if the gamma distribution for feature R
    N2 = np.random.normal(1, 0.5, dim) 
    Np = np.random.normal(0, 2, dim)
    Ny = np.random.normal(0, sy, dim)
    A = np.random.binomial(1, 0.5, size=dim) #0.5 is p_A indicating the proportion of individuals A=1

    # X var
    # Variable R defined as the salary
    R = N1 - l_h_r*A 
    # Variable R influence Q, a discrete variable that define a zone in a city
    R_A = 1/(1 + np.exp(l_r_q*R - l_h_q*A))
    Q = np.random.binomial(3, R_A) # this is a categorical variable

    # Y var
    # y target, with measurement and historical bias
    # Y var --> continuous var, denoted by s in the paper
    # y target, with measurement and historical bias
    if l_m_y_non_linear:
        # non-linear implementation of measurement bias on target y
        y = R - l_q*Q - l_y*A + l_m_y*A*(R<np.median(R)) - l_m_y*A*(R>=np.median(R)) + Ny + l_y_b*R*A # here I reversed the signs
    else:
        # linear implementation of measurement bias on target Y
        y = R - l_q*Q - l_y*A - l_m_y*A + Ny + l_y_b*R*A # adding measurement bias on y
    # y only historical, no measurement bias
    y_real = R - l_q*Q - l_y*A + Ny + l_y_b*R*A # this seems to be S of eq. 5d

    if l_m!=0:
      # Proxy for R, e.g. the dimension of the house
      P = R - A*l_m + Np # eq. 6a
      print("Correlation between R and P: ", np.corrcoef(P, R))
      dtf = pd.DataFrame({'P': P, 'Q':Q, 'A':A, 'Y':y, 'Y_real':y_real})
    else: 
      dtf = pd.DataFrame({'R':R, 'Q':Q, 'A':A, 'Y':y, 'Y_real':y_real})

    # Udersample
    int_p_u = int(((dtf['A']==1).sum())*p_u)
    if int_p_u > 0:
      if l_r:
          # Undersample with increasing R, for A=1 the people will results poor 
          drop_index = dtf.loc[dtf['A']==1, :].sort_values(by='R', ascending=True).index
          dtf = dtf.drop(drop_index[int_p_u:])
      else:
          dtf = dtf.drop(dtf.index[dtf['A']==1][int_p_u:])
    
    # Delete an important variable for omission: R or P, instead of delete I am saving it to compute the metrics later
    explainable_var = pd.Series()
    if l_o:
      if 'R' in dtf.columns:
        explainable_var = dtf.pop('R')
      elif 'P' in dtf.columns:
        explainable_var = dtf.pop('P')
      else:
        print("Condition non possible. How I could get here?")
    # Define feature matrix X and target Y
    X = dtf.reset_index(drop=True)

    # normalize
    df_num = X[[col for col in X.columns if (col!= 'Q') & (col!='A')]]
    scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
    X[df_num.columns] = scaled

    y_num = X['Y']
    y_real = X['Y_real']
    del X['Y']
    del X['Y_real']
    # Define threshold making y binary
    thres = y_num.mean()
    y = pd.Series(1*(y_num>thres))
    y_real = pd.Series(1*(y_real>thres))
    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=seed, stratify=X['A']==1)
    # individual set: does not have sensitive feature
    X_ind_train = X_train[[i for i in X_train.columns if i!='A']] 
    X_ind_test = X_test[[i for i in X_test.columns if i!='A']]
    # suppression set: does not have sensitive feature and at the same time the correlation between other features and the sensitive is lower than 1 (thr_supp) which is one for all scenarios
    X_supp_train = X_train[[i for i in X_train.columns if i!='A' and 
                            abs(np.corrcoef(X_train[i], X_train['A'])[0,1])<thr_supp]]
    X_supp_test = X_test[[i for i in X_test.columns if i!='A' and 
                          abs(np.corrcoef(X_train[i], X_train['A'])[0,1])<thr_supp]]
    # get y_test not biased
    y_train_real = y_real[y_train.index]
    y_test_real = y_real[y_test.index]

    X[['A','Q']] = X[['A','Q']].astype('object')

    metadata = {
                'dataset_name' : dataset_name,
                'prot1' :  'A',
                'prot2' : False,
                'target' : 'y',
                'pos_label' : 1,
                'neg_label' : 0,
                'R' : explainable_var, # for now
                'priv_group_A' : 1,
                'unpriv_group_A' : 0, # not sure, check paper
            }

    # replaced y_real with y_num to return 
    return X, y, y_num, X_train, X_ind_train, X_supp_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_train_real, y_test_real, explainable_var, metadata