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
      <td>very low</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0.000005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LyCoV555</td>
      <td>very low</td>
      <td>1</td>
      <td>0.166667</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LyCoV555</td>
      <td>very low</td>
      <td>1</td>
      <td>0.055556</td>
      <td>0.000058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LyCoV555</td>
      <td>very low</td>
      <td>1</td>
      <td>0.018519</td>
      <td>0.015295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LyCoV555</td>
      <td>very low</td>
      <td>1</td>
      <td>0.006173</td>
      <td>0.091381</td>
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
      <td>high</td>
      <td>2</td>
      <td>0.000686</td>
      <td>0.770867</td>
    </tr>
    <tr>
      <th>260</th>
      <td>4A8</td>
      <td>high</td>
      <td>2</td>
      <td>0.000229</td>
      <td>0.711955</td>
    </tr>
    <tr>
      <th>261</th>
      <td>4A8</td>
      <td>high</td>
      <td>2</td>
      <td>0.000076</td>
      <td>0.769362</td>
    </tr>
    <tr>
      <th>262</th>
      <td>4A8</td>
      <td>high</td>
      <td>2</td>
      <td>0.000025</td>
      <td>0.653560</td>
    </tr>
    <tr>
      <th>263</th>
      <td>4A8</td>
      <td>high</td>
      <td>2</td>
      <td>0.000008</td>
      <td>0.650991</td>
    </tr>
  </tbody>
</table>
<p>264 rows × 5 columns</p>
</div>



## Fit Hill curve to data using [`neutcurve`](https://jbloomlab.github.io/neutcurve/)


```python
fits = neutcurve.CurveFits(frac_infect, fixbottom= False, fixtop= False)
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
fitparams['virus'] = pd.Categorical(fitparams['virus'], categories=cat_order, ordered=True)
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
      <td>LyCoV555</td>
      <td>0.001076</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>929.396039</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LyCoV555</td>
      <td>0.001048</td>
      <td>interpolated</td>
      <td>low</td>
      <td>954.629329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LyCoV555</td>
      <td>0.001075</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>930.656270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LyCoV555</td>
      <td>0.004304</td>
      <td>interpolated</td>
      <td>high</td>
      <td>232.332685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S309</td>
      <td>0.012557</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>79.636579</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S309</td>
      <td>0.014038</td>
      <td>interpolated</td>
      <td>low</td>
      <td>71.237133</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S309</td>
      <td>0.016705</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>59.860800</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S309</td>
      <td>2.211712</td>
      <td>interpolated</td>
      <td>high</td>
      <td>0.452138</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4A8</td>
      <td>0.001612</td>
      <td>interpolated</td>
      <td>very low</td>
      <td>620.192223</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4A8</td>
      <td>0.001842</td>
      <td>interpolated</td>
      <td>low</td>
      <td>542.792735</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4A8</td>
      <td>0.001695</td>
      <td>interpolated</td>
      <td>medium</td>
      <td>590.119880</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4A8</td>
      <td>0.410878</td>
      <td>interpolated</td>
      <td>high</td>
      <td>2.433810</td>
    </tr>
  </tbody>
</table>
</div>




```python
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

```

## Plot neut curves for mAbs


```python
#not sure about color, but at least I managed to change them! When I indicate colors this way, does it pick from the cbpalatte?
fig, axes = fits.plotSera(
                          viruses=['very low', 'low', 'medium', 'high'],
                          xlabel='concentration (ug/ml)',
                          ncol=6,
                          widthscale=2,
                          heightscale=2,
                          titlesize=28, labelsize=28, ticksize=18, legendfontsize=24, yticklocs=[0,0.5,1],
                          markersize=8, linewidth=2,
                          virus_to_color_marker={
                              'very low': ('red', 'o'),
                              'low': ('blue', '^'),
                              'medium': ('purple', 's'),
                              'high': ('black', 'd')},
                          sharex=False
                         )
```


    
![png](virus_neutralization_mAbs_files/virus_neutralization_mAbs_19_0.png)
    



```python

```
