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
frac_infect = pd.read_csv(config['mAb_neuts'], index_col=0)
```


```python
frac_infect['serum'] = frac_infect['serum'].map({'LyCoV555': 'LY-CoV555 (RBD class 2)',
                                                 'S309': 'S309 (RBD class 3)',
                                                 '4A8': '4A8 (NTD)'})

```


```python
frac_infect
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>very low</td>
      <td>1</td>
      <td>0.166667</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>very low</td>
      <td>1</td>
      <td>0.041667</td>
      <td>0.003510</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>very low</td>
      <td>1</td>
      <td>0.010417</td>
      <td>0.099252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>very low</td>
      <td>1</td>
      <td>0.002604</td>
      <td>0.513108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>very low</td>
      <td>1</td>
      <td>0.000651</td>
      <td>0.850309</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>187</th>
      <td>4A8 (NTD)</td>
      <td>high</td>
      <td>2</td>
      <td>0.002604</td>
      <td>1.140743</td>
    </tr>
    <tr>
      <th>188</th>
      <td>4A8 (NTD)</td>
      <td>high</td>
      <td>2</td>
      <td>0.000651</td>
      <td>0.979990</td>
    </tr>
    <tr>
      <th>189</th>
      <td>4A8 (NTD)</td>
      <td>high</td>
      <td>2</td>
      <td>0.000163</td>
      <td>0.950808</td>
    </tr>
    <tr>
      <th>190</th>
      <td>4A8 (NTD)</td>
      <td>high</td>
      <td>2</td>
      <td>0.000041</td>
      <td>0.985863</td>
    </tr>
    <tr>
      <th>191</th>
      <td>4A8 (NTD)</td>
      <td>high</td>
      <td>2</td>
      <td>0.000010</td>
      <td>0.923761</td>
    </tr>
  </tbody>
</table>
<p>192 rows × 5 columns</p>
</div>



## Fit Hill curve to data using [`neutcurve`](https://jbloomlab.github.io/neutcurve/)


```python
fits = neutcurve.CurveFits(frac_infect, fixbottom= False, fixtop= True)
```


```python
fitparams = (
        fits.fitParams()
        # get columns of interest
        [['serum', 'ic50', 'ic50_bound','virus']]
        .assign(NT50=lambda x: 1/x['ic50'])        
        )
```


```python
cat_order = ['very low', 'low', 'medium', 'high']
fitparams['virus'] = pd.Categorical(fitparams['virus'],
                                    categories=cat_order,
                                    ordered=True)
```


```python
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
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>virus</th>
      <th>NT50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>0.002520</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>396.764998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>0.003043</td>
      <td>interpolated</td>
      <td>low</td>
      <td>328.656981</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>0.004048</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>247.026726</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LY-CoV555 (RBD class 2)</td>
      <td>0.015075</td>
      <td>interpolated</td>
      <td>high</td>
      <td>66.334383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S309 (RBD class 3)</td>
      <td>0.011144</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>89.731906</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S309 (RBD class 3)</td>
      <td>0.022513</td>
      <td>interpolated</td>
      <td>low</td>
      <td>44.419146</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S309 (RBD class 3)</td>
      <td>0.031744</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>31.501638</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S309 (RBD class 3)</td>
      <td>6.000000</td>
      <td>lower</td>
      <td>high</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4A8 (NTD)</td>
      <td>0.005148</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>194.232873</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4A8 (NTD)</td>
      <td>0.006607</td>
      <td>interpolated</td>
      <td>low</td>
      <td>151.363779</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4A8 (NTD)</td>
      <td>0.007905</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>126.494615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4A8 (NTD)</td>
      <td>0.166667</td>
      <td>lower</td>
      <td>high</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

```

## Plot neut curves for mAbs


```python
fig, axes = fits.plotSera(
                          viruses=['very low', 'low', 'medium', 'high'],
                          xlabel='concentration (µg/ml)',
                          ncol=3,
                          widthscale=1,
                          heightscale=1.2,
                          titlesize=14, labelsize=14, ticksize=11,
                          legendfontsize=14, yticklocs=[0,0.5,1],
                          markersize=5, linewidth=1,
                          legendtitle='Cell ACE2 expression' ,
                          virus_to_color_marker={
                              'very low': ('#F0E442', 'o'),
                              'low': ('#CC79A7', 'o'),
                              'medium': ('#009E73', 'o'),
                              'high': ('#0072B2', 'o')},
                          sharex=False
                         )
```


    
![png](virus_neutralization_mAbs_files/virus_neutralization_mAbs_20_0.png)
    



```python

```
