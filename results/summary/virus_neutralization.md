# Analysis of SARS-COV-2 virus neutalization in different Ace2 clones

This notebook analysis neutralization of SARS-COV-2/Wu-1 virus by sera from vaccinated individuals on 293T cell clones that express different levels of ACE2.

### Set up Analysis


```python
import os
import warnings

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from plotnine import *
import seaborn

import neutcurve
from neutcurve.colorschemes import CBMARKERS, CBPALETTE

import yaml
```


```python
warnings.simplefilter('ignore')
```

Read config file.


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Set seaborn theme:


```python
theme_set(theme_seaborn(style='white', context='talk', font_scale=1))
plt.style.use('seaborn-white')
```


```python
resultsdir=config['resultsdir']
os.makedirs(resultsdir, exist_ok=True)
```

## Read in data
We read in fraction infectivirty data for different cell lines and import sera information.


```python
frac_infect = list() # create df list

for f in config['depletion_neuts'].keys():
    df = (pd.read_csv(f, index_col=0))
    frac_infect.append(df)  
frac_infect = pd.concat(frac_infect)

frac_infect['serum'] = frac_infect['serum'] + '__' + frac_infect['cells']

frac_infect['virus'] = frac_infect['virus'].str.replace('post-depletion','depleted')
frac_infect['virus'] = frac_infect['virus'].str.replace('pre-depletion','not depleted')

```


```python
#read in sample info
sample_information = (pd.read_csv(config['sample_information'])
                      .drop_duplicates())

sample_information['sorted']=sample_information['subject_name'].str[:-1].astype(int)
sample_information = sample_information.sort_values('sorted')

#store sera names in a list to later convert to factors for plotting
cat_order_sera = sample_information['serum'].tolist()

```


```python
#read in ACE2 expression info
ACE2_expression_df = (pd.read_csv(config['ACE2_expression_df'])
                      .drop_duplicates())
```

## Fit Hill curve 

We use [`neutcurve`](https://jbloomlab.github.io/neutcurve/) to fit Hill curve for neutralization data and calcualte IC50 and NT50 values.


```python
frac_infect.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serum</th>
      <th>virus</th>
      <th>replicate</th>
      <th>concentration</th>
      <th>fraction infectivity</th>
      <th>cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>1</td>
      <td>0.040000</td>
      <td>0.000018</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>1</td>
      <td>0.010000</td>
      <td>0.000027</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>1</td>
      <td>0.002500</td>
      <td>-0.000010</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>1</td>
      <td>0.000625</td>
      <td>-0.000006</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>1</td>
      <td>0.000156</td>
      <td>0.058871</td>
      <td>very low</td>
    </tr>
  </tbody>
</table>
</div>




```python
fits = neutcurve.CurveFits(frac_infect, fixbottom=0)

fitparams = (
    fits.fitParams()
    .rename(columns={'virus': 'RBD-targeting antibodies'})
    [['serum', 'RBD-targeting antibodies', 'ic50', 'ic50_bound']]
    .assign(NT50=lambda x: 1/x['ic50'])

    )
```


```python
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

```


```python
fitparams[['sample', 'cells']] = fitparams['serum'].str.split('__', 1, expand=True)
```


```python
#category for cell order so that ggplot does not use alphabetical
cat_order = ['very low', 'low', 'medium', 'high']
fitparams['cells'] = pd.Categorical(fitparams['cells'], categories=cat_order, ordered=True)

fitparams['sample'] = pd.Categorical(fitparams['sample'], categories=cat_order_sera, ordered=True)
```


```python
#save data
fitparams.to_csv(config['neuts'], index=False)
```


```python
# add % RBD-targetting antibodies
df_pre = fitparams.loc[fitparams['RBD-targeting antibodies'] == 'not depleted']
df_post = fitparams.loc[fitparams['RBD-targeting antibodies'] == 'depleted']
df_mege = pd.merge(df_pre, df_post, on="serum")
df_mege['percent_RBD'] = (df_mege['NT50_x']-df_mege['NT50_y'])/df_mege['NT50_x']*100
df_mege['percent_RBD'] = df_mege['percent_RBD'].astype(int)
fitparams = pd.merge(fitparams,df_mege[['serum','percent_RBD']],on='serum', how='left')
fitparams['percent_RBD_str'] = fitparams['percent_RBD'].astype(str) +'%'
fitparams
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serum</th>
      <th>RBD-targeting antibodies</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>NT50</th>
      <th>ic50_is_bound</th>
      <th>sample</th>
      <th>cells</th>
      <th>percent_RBD</th>
      <th>percent_RBD_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>0.000036</td>
      <td>interpolated</td>
      <td>27704.327914</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>very low</td>
      <td>95</td>
      <td>95%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63C-day-10__very low</td>
      <td>depleted</td>
      <td>0.000896</td>
      <td>interpolated</td>
      <td>1116.350361</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>very low</td>
      <td>95</td>
      <td>95%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64C-day-15__very low</td>
      <td>not depleted</td>
      <td>0.000058</td>
      <td>interpolated</td>
      <td>17250.386352</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>very low</td>
      <td>86</td>
      <td>86%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64C-day-15__very low</td>
      <td>depleted</td>
      <td>0.000416</td>
      <td>interpolated</td>
      <td>2404.654223</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>very low</td>
      <td>86</td>
      <td>86%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99C-day-27__very low</td>
      <td>not depleted</td>
      <td>0.000095</td>
      <td>interpolated</td>
      <td>10482.828819</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>very low</td>
      <td>83</td>
      <td>83%</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>194C-day-8__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
      <td>194C-day-8</td>
      <td>high</td>
      <td>99</td>
      <td>99%</td>
    </tr>
    <tr>
      <th>76</th>
      <td>215C-day-19__high</td>
      <td>not depleted</td>
      <td>0.000550</td>
      <td>interpolated</td>
      <td>1817.157984</td>
      <td>False</td>
      <td>215C-day-19</td>
      <td>high</td>
      <td>98</td>
      <td>98%</td>
    </tr>
    <tr>
      <th>77</th>
      <td>215C-day-19__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
      <td>215C-day-19</td>
      <td>high</td>
      <td>98</td>
      <td>98%</td>
    </tr>
    <tr>
      <th>78</th>
      <td>229C-day-29__high</td>
      <td>not depleted</td>
      <td>0.000301</td>
      <td>interpolated</td>
      <td>3317.975700</td>
      <td>False</td>
      <td>229C-day-29</td>
      <td>high</td>
      <td>99</td>
      <td>99%</td>
    </tr>
    <tr>
      <th>79</th>
      <td>229C-day-29__high</td>
      <td>depleted</td>
      <td>0.039109</td>
      <td>interpolated</td>
      <td>25.569526</td>
      <td>False</td>
      <td>229C-day-29</td>
      <td>high</td>
      <td>99</td>
      <td>99%</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 10 columns</p>
</div>




```python
rbd = fitparams.loc[fitparams['RBD-targeting antibodies'] == 'depleted']
```

## Plot IC50 values


```python
IC50 = (ggplot(fitparams, aes(x='cells',
                              y='ic50',
                              colour='RBD-targeting antibodies',
                              group = 'RBD-targeting antibodies',
                              )) +
              geom_point(size=3) +
              geom_line(alpha=1) +
         geom_text(rbd, aes(label = 'percent_RBD_str',
                            y=rbd['ic50'].max()*1.5),
                  colour = CBPALETTE[0]) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=12),
                   axis_text_x=element_text(size=12, angle= 45),
                   legend_text=element_text(size=12),
                   legend_title=element_text(size=12),
                   axis_title_x=element_text(size=18),
                   strip_text = element_text(size=12)
                  ) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='Inhibitory Concentration 50%') +
              xlab('ACE2 expression in target cells') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = IC50.draw()
```


    
![png](virus_neutralization_files/virus_neutralization_24_0.png)
    


## Plot NT50 values


```python
NT50 = (ggplot(fitparams, aes(x='cells', y='NT50', colour='RBD-targeting antibodies', group = 'RBD-targeting antibodies')) +
              geom_point(size=3) +
             geom_line(alpha=1) +
             geom_text(rbd, aes(label = 'percent_RBD_str',
                        y=rbd['NT50'].max()*12),
              colour = CBPALETTE[0]) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=12),
                   axis_text_x=element_text(size=12, angle= 45),
                   legend_text=element_text(size=12),
                   legend_title=element_text(size=12),
                   axis_title_x=element_text(size=18),
                   strip_text = element_text(size=12)
                  ) +
                geom_hline(yintercept=config['NT50_LOD'], 
                linetype='dotted', 
                size=1, 
                alpha=0.6, 
                color=CBPALETTE[7]) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='Neutralization Titer (NT50)') +
              xlab('ACE2 expression in target cells') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = NT50.draw()
```


    
![png](virus_neutralization_files/virus_neutralization_26_0.png)
    



```python
df_merged = pd.merge(fitparams, ACE2_expression_df, on='cells')

```


```python
plot = (
    ggplot(df_merged) +
    aes(x="relative MFI", y="NT50", group='RBD-targeting antibodies', color='RBD-targeting antibodies')+
    geom_point(size=2) +
    geom_line() +
    theme(figure_size=(16,6),
          axis_ticks_minor_x=None,
          axis_text=element_text(size=14),
          axis_text_x=element_text(size=14),
          legend_text=element_text(size=14),
          legend_title=element_text(size=12),
          axis_title_x=element_text(size=18),
          strip_text = element_text(size=14)) +
    scale_color_manual(values= ['#56B4E9','#E69F00']) +
    labs(title="Neutralization Titer vs ACE2 Expression", x="ACE2 expression relative\nto highest ACE2 cells", y="Neutralization Titer (NT50)") +
    scale_x_log10() +
    scale_y_log10() +
    facet_wrap('sample', ncol=5) +
    geom_hline(yintercept=25,
                linetype='dotted', 
                size=1, 
                alpha=0.6, 
                color=CBPALETTE[7])
   

)

plot.draw()
plot.save(f'{resultsdir}/NT50_vs_ACE2_expression.pdf')
```


    
![png](virus_neutralization_files/virus_neutralization_28_0.png)
    


## Plot neut curves for all samples


```python
fig, axes = fits.plotSera(
                          xlabel='serum dilution',
                          ncol=8,
                          widthscale=2.5,
                          heightscale=2.5,
                          titlesize=55, labelsize=55, ticksize=45, legendfontsize=55, yticklocs=[0,0.5,1],
                          markersize=10, linewidth=6,
                          virus_to_color_marker={
                          'depleted': ('#56B4E9', 'o'),
                          'not depleted': ('#E69F00', 'o')},
                          legendtitle='RBD-targeting antibodies',
                    
                         )
```


    
![png](virus_neutralization_files/virus_neutralization_30_0.png)
    



```python

```


```python

```


```python

```
