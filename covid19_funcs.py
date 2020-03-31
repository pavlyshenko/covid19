# Predictive Analytics for Coronavirus Spread Using Bayesian Inference
# Author: Bohdan Pavlyshenko
# LinkedIn: https://www.linkedin.com/in/bpavlyshenko/  
# GitHub: https://github.com/pavlyshenko/coronavirus

import pystan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
import pystan
import datetime
import pickle
sns.set()

# Set up options 

# fields names:
region_field, cases_field,fatalities_field='region','cases','deaths'

# Directory for images
img_dir='imgs'

# Normalization coefficients
target_field_norm_coef=1/100000
time_var_norm_coef=1/7

# Model file names
model_file='model.pkl'

# fields names:
region_field, cases_field,fatalities_field='region','cases','deaths'

# Directory for images
img_dir='imgs'

######################################

model_logistic = """
    data {
        int<lower=1> n;
        int<lower=1> n_pred;
        vector[n] y;
        vector[n] t;
        vector[n_pred] t_pred;
    }
    parameters {
        real<lower=0> alpha;
        real<lower=0> beta;
        real<lower=0> t0;
        real<lower=0> sigma; 
    }
    model {
    alpha~normal(1,1);
    beta~normal(1,1);
    t0~normal(10,10);
    y ~ normal(alpha ./ (1 + exp(-(beta*(t-t0)))), sigma);
    }
    generated quantities {
      vector[n_pred] pred;
      for (i in 1:n_pred)
      pred[i] = normal_rng(alpha / (1 + exp(-(beta*(t_pred[i]-t0)))),sigma);
    }
    """
def compile_model(model_name,model_file):
    stan_model= pystan.StanModel(model_code=model_name)
    with open(model_file, 'wb') as f:
        pickle.dump(stan_model, f)
    print ('Done')
    
def plot_results(df_res,target_field,region_value, fit_samples, filename='p1.png'):
    fig, ax = plt.subplots(2,2, sharex=False, figsize=(15,10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    df_res['pred'].plot(yerr=df_res['pred_yerr'].values,title='Prediction ($10^5$)',ecolor='#aaaaaa',ax=ax[0,0])
    ax[0,0].scatter(df_res.index,df_res.y,s=10,c='green')
    df_res[['n_per_day_real','n_per_day_prediction']]=df_res[['y','pred']]-df_res[['y','pred']].shift(1)
    df_res[['n_per_day_real','n_per_day_prediction']].plot(ax = ax[0,1],title='Number per day ($10^5$)')
    alpha_mean=np.round(fit_samples['alpha'].mean(),3)
    alpha_std=np.round(fit_samples['alpha'].std(),3)
    beta_mean=np.round(fit_samples['beta'].mean(),3)
    beta_std=np.round(fit_samples['beta'].std(),3)
    t0_mean=np.round(fit_samples['t0'].mean(),3)
    t0_std=np.round(fit_samples['t0'].std(),3)
    print (f'Model parameters: alpha={alpha_mean} (sd:{alpha_std}, beta={beta_mean}(sd:{beta_std}),\
    t0={t0_mean}(sd:{t0_std})')
    alpha_samples=np.round(pd.Series(fit_samples['alpha']),3)
    alpha_samples.plot(kind='density', 
    title=r'$\alpha$'+f' (mean:{alpha_mean}, sd:{alpha_std})',ax = ax[1,0])
    beta_samples=pd.Series(fit_samples['beta'])
    beta_samples.plot(kind='density', 
    title=r'$\beta$'+f' (mean:{beta_mean}, sd:{beta_std})',ax = ax[1,1])
    fig.suptitle(f'Number of {target_field} for {region_value}')
    plt.savefig(filename, dpi=150,bbox_inches='tight')
    plt.show()
    
def get_prediction(df,n_days_predict=25, target_field='cases',
                   region_field='region', region_value='China',
                   target_field_norm_coef=target_field_norm_coef,
                   time_var_norm_coef=time_var_norm_coef, model_file=model_file,
                   img_file_name='p1.png'):
    df.date=pd.to_datetime(df.date)
    df_res=df.loc[df[region_field]==region_value, ['date',target_field]].set_index('date')\
    .groupby(pd.Grouper(freq='D'))\
    [target_field].sum().to_frame('y').reset_index()
    print ('Time Series size:',df_res.shape[0])
    n_train=df_res.shape[0]
    maxdate=df_res.date.max()
    for i in np.arange(1,n_days_predict+1):
        df_res=df_res.append(pd.DataFrame({'date':\
            [maxdate+datetime.timedelta(days=int(i))]}))
    df_res['t']=time_var_norm_coef*np.arange(df_res.shape[0])
    df_res.y=target_field_norm_coef*df_res.y
    df_res.set_index('date',inplace=True)
    data = {'n': n_train,'n_pred':df_res.shape[0],
            'y': df_res.iloc[:n_train,:].y.values,'t':df_res.iloc[:n_train,:]\
            .t.values,'t_pred':df_res.t.values}

    stan_model = pickle.load(open('model.pkl', 'rb'))
    fit=stan_model.sampling(data=data, iter=5000, chains=3)

    fit_samples = fit.extract(permuted=True)
    pred=fit_samples['pred']
    df_res['pred']=pred.mean(axis=0)
    df_res['pred_yerr']=(pd.DataFrame(pred).quantile(q=0.95,axis=0).values-\
                       pd.DataFrame(pred).quantile(q=0.05,axis=0).values)/2
    plot_results(df_res,target_field,region_value,fit_samples, img_file_name)
    return(df_res)
    
def get_regions_prediction(df,region_list,region_field=region_field,target_field='cases',img_dir=img_dir):
    for i in region_list:
        print (f'\n{i}:')
        df_res=get_prediction(df,n_days_predict=25, target_field=target_field,region_value=i)