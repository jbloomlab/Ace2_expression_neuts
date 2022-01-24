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
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>6.000000</td>
      <td>0.000047</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>2.000000</td>
      <td>0.000037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.666667</td>
      <td>0.000803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.222222</td>
      <td>0.000670</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.074074</td>
      <td>0.008842</td>
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
      <th>259</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.008230</td>
      <td>0.163645</td>
    </tr>
    <tr>
      <th>260</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.002743</td>
      <td>0.296861</td>
    </tr>
    <tr>
      <th>261</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000914</td>
      <td>0.528512</td>
    </tr>
    <tr>
      <th>262</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000305</td>
      <td>0.613362</td>
    </tr>
    <tr>
      <th>263</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000102</td>
      <td>0.642189</td>
    </tr>
  </tbody>
</table>
<p>264 rows × 5 columns</p>
</div>



## Fit Hill curve to data using [`neutcurve`](https://jbloomlab.github.io/neutcurve/)


```python
fits = neutcurve.CurveFits(frac_infect)
```


```python
fitparams = (
        fits.fitParams()
        # get columns of interest
        [['serum', 'ic50', 'ic50_bound']]
        .assign(NT50=lambda x: 1/x['ic50'])        
        )
```


```python
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

```

## Plot neut curves for mAbs


```python
fig, axes = fits.plotSera(
                          xlabel='serum dilution',
                          ncol=6,
                          widthscale=2,
                          heightscale=2,
                          titlesize=12, labelsize=24, ticksize=15, legendfontsize=24, yticklocs=[0,0.5,1],
                          markersize=8, linewidth=2,
                         )
```


    
![png](virus_neutralization_mAbs_files/virus_neutralization_mAbs_17_0.png)
    

