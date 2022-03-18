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
      <td>0.500000</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.166667</td>
      <td>0.000992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.055556</td>
      <td>0.012796</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.018519</td>
      <td>0.096506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LyCoV555</td>
      <td>consensus_Kozak</td>
      <td>1</td>
      <td>0.006173</td>
      <td>0.356711</td>
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
      <td>0.000686</td>
      <td>0.672363</td>
    </tr>
    <tr>
      <th>260</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000229</td>
      <td>0.789911</td>
    </tr>
    <tr>
      <th>261</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000076</td>
      <td>0.775086</td>
    </tr>
    <tr>
      <th>262</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000025</td>
      <td>0.888502</td>
    </tr>
    <tr>
      <th>263</th>
      <td>4A8</td>
      <td>clone_A</td>
      <td>2</td>
      <td>0.000008</td>
      <td>0.838611</td>
    </tr>
  </tbody>
</table>
<p>264 rows × 5 columns</p>
</div>



## Fit Hill curve to data using [`neutcurve`](https://jbloomlab.github.io/neutcurve/)


```python
fits = neutcurve.CurveFits(frac_infect, fixbottom= False, fixtop= False
                           )
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
                          titlesize=20, labelsize=24, ticksize=15, legendfontsize=24, yticklocs=[0,0.5,1],
                          markersize=8, linewidth=2,
                         )
```


    
![png](virus_neutralization_mAbs_files/virus_neutralization_mAbs_17_0.png)
    



```python

```
