# Analysis of SARS-COV-2 virus neutalization in different Ace2 clones

### Set up Analysis


```python
import itertools
import math
import os
import re
import warnings

from IPython.display import display, HTML

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort

import numpy as np
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


```python
sample_information = (pd.read_csv(config['sample_information'])
                      .drop_duplicates()
                     )

frac_infect = list() # create df list

for f in config['depletion_neuts'].keys():
    df = (pd.read_csv(f, index_col=0).assign(cells=config['depletion_neuts'][f]))
    df = df.merge(sample_information, on='serum')
    frac_infect.append(df)  
    
```


```python
for df in frac_infect:
    df['serum'] = df['serum'] + '__' + df['cells']
```


```python
sample_information
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
      <th>day</th>
      <th>age</th>
      <th>vaccine</th>
      <th>subject_name</th>
      <th>serum</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>18-55y</td>
      <td>Pfizer</td>
      <td>192C</td>
      <td>192C-day-9</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>18-55y</td>
      <td>Pfizer</td>
      <td>194C</td>
      <td>194C-day-8</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>18-55y</td>
      <td>Pfizer</td>
      <td>63C</td>
      <td>63C-day-10</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>&gt;55y</td>
      <td>Pfizer</td>
      <td>64C</td>
      <td>64C-day-15</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>18-55y</td>
      <td>Pfizer</td>
      <td>99C</td>
      <td>99C-day-27</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>18</td>
      <td>&gt;55y</td>
      <td>Pfizer</td>
      <td>108C</td>
      <td>108C-day-18</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



## Fit Hill curve to data using [`neutcurve`](https://jbloomlab.github.io/neutcurve/)


```python
frac_infect_combined = list()

for cells in frac_infect:
    fits = neutcurve.CurveFits(cells, fixbottom= False)
    frac_infect_combined.append(fits)
```


```python
fitparams_combined = pd.DataFrame() # create empty data frame

for fits in frac_infect_combined:
    fitparams = (
        fits.fitParams()
        .rename(columns={'virus': 'depletion'})
        # get columns of interest
        [['serum', 'depletion', 'ic50', 'ic50_bound']]
        .assign(NT50=lambda x: 1/x['ic50'])
#         .merge(sample_information, on=['serum'])
        
        )
    fitparams_combined = fitparams_combined.append(fitparams).reset_index(drop=True)
    

fitparams_combined['ic50_is_bound'] = fitparams_combined['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

fitparams_combined
fitparams_combined.to_csv(config['neuts'], index=False)

```


```python
fitparams_combined
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
      <th>depletion</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>NT50</th>
      <th>ic50_is_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63C-day-10__very_low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000031</td>
      <td>interpolated</td>
      <td>31968.943613</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63C-day-10__very_low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000454</td>
      <td>interpolated</td>
      <td>2200.799322</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64C-day-15__very_low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000064</td>
      <td>interpolated</td>
      <td>15594.993852</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64C-day-15__very_low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000199</td>
      <td>interpolated</td>
      <td>5029.538042</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99C-day-27__very_low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000096</td>
      <td>interpolated</td>
      <td>10374.149491</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>99C-day-27__very_low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000435</td>
      <td>interpolated</td>
      <td>2296.488086</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>108C-day-18__very_low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000038</td>
      <td>interpolated</td>
      <td>26336.105034</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>108C-day-18__very_low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000225</td>
      <td>interpolated</td>
      <td>4450.299723</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>63C-day-10__low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000040</td>
      <td>interpolated</td>
      <td>24852.475407</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>63C-day-10__low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000704</td>
      <td>interpolated</td>
      <td>1420.647038</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>64C-day-15__low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000081</td>
      <td>interpolated</td>
      <td>12328.306795</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>64C-day-15__low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000262</td>
      <td>interpolated</td>
      <td>3823.459538</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>99C-day-27__low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000110</td>
      <td>interpolated</td>
      <td>9071.489070</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>99C-day-27__low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000418</td>
      <td>interpolated</td>
      <td>2392.117387</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>108C-day-18__low_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000065</td>
      <td>interpolated</td>
      <td>15482.242047</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>108C-day-18__low_ACE2</td>
      <td>post-depletion</td>
      <td>0.000487</td>
      <td>interpolated</td>
      <td>2052.442693</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>63C-day-10__high_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000102</td>
      <td>interpolated</td>
      <td>9830.591738</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>63C-day-10__high_ACE2</td>
      <td>post-depletion</td>
      <td>0.012598</td>
      <td>interpolated</td>
      <td>79.377273</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>64C-day-15__high_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2656.009243</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>64C-day-15__high_ACE2</td>
      <td>post-depletion</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>99C-day-27__high_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2655.584701</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>99C-day-27__high_ACE2</td>
      <td>post-depletion</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>108C-day-18__high_ACE2</td>
      <td>pre-depletion</td>
      <td>0.000208</td>
      <td>interpolated</td>
      <td>4814.580041</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>108C-day-18__high_ACE2</td>
      <td>post-depletion</td>
      <td>0.010558</td>
      <td>interpolated</td>
      <td>94.715361</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
fitparams_combined[['sample', 'cells']] = fitparams_combined['serum'].str.split('__', 1, expand=True)

```

## Plot IC50 values


```python
IC50 = (ggplot(fitparams_combined, aes(x='cells', y='ic50', colour='depletion')) +
              geom_point(size=3) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=10),
                   axis_text_x=element_text(size=10, angle= 45),
                   legend_text=element_text(size=10),
                   legend_title=element_text(size=10),
                   axis_title_x=element_text(size=10),
                   strip_text = element_text(size=10)
                  ) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='IC50') +
              xlab('cell clone') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = IC50.draw()
# IC50.save(f'./{resultsdir}/IC50.pdf')
```


    
![png](virus_neutralization_files/virus_neutralization_19_0.png)
    


## Plot NT50 values


```python
NT50 = (ggplot(fitparams_combined, aes(x='cells', y='NT50', colour='depletion')) +
              geom_point(size=3) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=10),
                   axis_text_x=element_text(size=10, angle= 45),
                   legend_text=element_text(size=10),
                   legend_title=element_text(size=10),
                   axis_title_x=element_text(size=10),
                   strip_text = element_text(size=10)
                  ) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='NT50') +
              xlab('cell clone') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = NT50.draw()
# NT50.save(f'./{resultsdir}/IC50.pdf')

```


    
![png](virus_neutralization_files/virus_neutralization_21_0.png)
    


## IC50 fold change


```python
df_pre = fitparams_combined.loc[fitparams_combined['depletion'] == 'pre-depletion']
df_post = fitparams_combined.loc[fitparams_combined['depletion'] == 'post-depletion']
df_mege = pd.merge(df_pre, df_post, on="serum")
df_mege['IC50_fold_change'] = df_mege['ic50_x']/df_mege['ic50_y']
```


```python
IC50_fc = (ggplot(df_mege, aes(x='cells_y', y='IC50_fold_change')) +
              geom_point(size=3) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=10),
                   axis_text_x=element_text(size=10, angle= 45),
                   legend_text=element_text(size=10),
                   legend_title=element_text(size=10),
                   axis_title_x=element_text(size=10),
                   axis_title_y=element_text(size=10),
                   strip_text = element_text(size=10)
                  ) +
              facet_wrap('sample_y', ncol = 4)+
              scale_y_log10(name='IC50 fold change (pre-depletion/post-depletion)') +
              xlab('cell clone') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = IC50_fc.draw()
# IC50.save(f'./{resultsdir}/IC50.pdf')
```


    
![png](virus_neutralization_files/virus_neutralization_24_0.png)
    


## Make horizontal line plot connecting pre- and post-IC50


```python
p = (ggplot(fitparams_combined, 
            aes(x='NT50',
                y='cells',
                fill='depletion',
                group='cells',
               )) +
     scale_x_log10(name='neutralization titer 50% (NT50)', 
                   limits=[config['NT50_LOD'],fitparams_combined['NT50'].max()*3]) +
     geom_vline(xintercept=config['NT50_LOD'], 
                linetype='dotted', 
                size=1, 
                alpha=0.6, 
                color=CBPALETTE[7]) +
     geom_line(alpha=1, color=CBPALETTE[0]) +
     geom_point(size=4, color=CBPALETTE[0]) +
     theme(figure_size=(15,1*df['serum'].nunique()),
           axis_text=element_text(size=12),
           legend_text=element_text(size=12),
           legend_title=element_text(size=12),
           axis_title_x=element_text(size=12),
           strip_text = element_text(size=12)
          ) +
     facet_wrap('sample', ncol = 4) +
     ylab('') +
    scale_fill_manual(values=['#E69F00', '#56B4E9', ], 
                     name='pre- or post-depletion\nof RBD antibodies')
    )

_ = p.draw()

```


    
![png](virus_neutralization_files/virus_neutralization_26_0.png)
    


## Plot neut curves for all samples


```python
for fits in frac_infect_combined:
    fig, axes = fits.plotSera(
                              xlabel='serum dilution',
                              ncol=4,
                              widthscale=2,
                              heightscale=2,
                              titlesize=20, labelsize=24, ticksize=15, legendfontsize=24, yticklocs=[0,0.5,1],
                              markersize=8, linewidth=2,
                              virus_to_color_marker={
                              'pre-depletion': ('#56B4E9', 'o'),
                              'post-depletion': ('#E69F00', 'o')}
                             )
```


    
![png](virus_neutralization_files/virus_neutralization_28_0.png)
    



    
![png](virus_neutralization_files/virus_neutralization_28_1.png)
    



    
![png](virus_neutralization_files/virus_neutralization_28_2.png)
    



```python

```
