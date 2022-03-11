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

## Fit Hill curve 

We use [`neutcurve`](https://jbloomlab.github.io/neutcurve/) to fit Hill curve for neutralization data and calcualte IC50 and NT50 values.


```python
fits = neutcurve.CurveFits(frac_infect, fixbottom= False)

fitparams = (
    fits.fitParams()
    .rename(columns={'virus': 'RBD-targeting antibodies'})
    [['serum', 'RBD-targeting antibodies', 'ic50', 'ic50_bound']]
    .assign(NT50=lambda x: 1/x['ic50'])

    )
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
      <th>RBD-targeting antibodies</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>NT50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>0.000031</td>
      <td>interpolated</td>
      <td>31968.943613</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63C-day-10__very low</td>
      <td>depleted</td>
      <td>0.000454</td>
      <td>interpolated</td>
      <td>2200.799322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64C-day-15__very low</td>
      <td>not depleted</td>
      <td>0.000064</td>
      <td>interpolated</td>
      <td>15594.993852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64C-day-15__very low</td>
      <td>depleted</td>
      <td>0.000199</td>
      <td>interpolated</td>
      <td>5029.538042</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99C-day-27__very low</td>
      <td>not depleted</td>
      <td>0.000096</td>
      <td>interpolated</td>
      <td>10374.149491</td>
    </tr>
    <tr>
      <th>5</th>
      <td>99C-day-27__very low</td>
      <td>depleted</td>
      <td>0.000435</td>
      <td>interpolated</td>
      <td>2296.488086</td>
    </tr>
    <tr>
      <th>6</th>
      <td>108C-day-18__very low</td>
      <td>not depleted</td>
      <td>0.000038</td>
      <td>interpolated</td>
      <td>26336.105034</td>
    </tr>
    <tr>
      <th>7</th>
      <td>108C-day-18__very low</td>
      <td>depleted</td>
      <td>0.000225</td>
      <td>interpolated</td>
      <td>4450.299723</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120C-day-10__very low</td>
      <td>not depleted</td>
      <td>0.000027</td>
      <td>interpolated</td>
      <td>36527.531964</td>
    </tr>
    <tr>
      <th>9</th>
      <td>120C-day-10__very low</td>
      <td>depleted</td>
      <td>0.000218</td>
      <td>interpolated</td>
      <td>4588.546268</td>
    </tr>
    <tr>
      <th>10</th>
      <td>180C-day-36__very low</td>
      <td>not depleted</td>
      <td>0.000048</td>
      <td>interpolated</td>
      <td>21040.240657</td>
    </tr>
    <tr>
      <th>11</th>
      <td>180C-day-36__very low</td>
      <td>depleted</td>
      <td>0.000301</td>
      <td>interpolated</td>
      <td>3322.323929</td>
    </tr>
    <tr>
      <th>12</th>
      <td>63C-day-10__low</td>
      <td>not depleted</td>
      <td>0.000040</td>
      <td>interpolated</td>
      <td>24852.475407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>63C-day-10__low</td>
      <td>depleted</td>
      <td>0.000704</td>
      <td>interpolated</td>
      <td>1420.647038</td>
    </tr>
    <tr>
      <th>14</th>
      <td>64C-day-15__low</td>
      <td>not depleted</td>
      <td>0.000081</td>
      <td>interpolated</td>
      <td>12328.306795</td>
    </tr>
    <tr>
      <th>15</th>
      <td>64C-day-15__low</td>
      <td>depleted</td>
      <td>0.000262</td>
      <td>interpolated</td>
      <td>3823.459538</td>
    </tr>
    <tr>
      <th>16</th>
      <td>99C-day-27__low</td>
      <td>not depleted</td>
      <td>0.000110</td>
      <td>interpolated</td>
      <td>9071.489070</td>
    </tr>
    <tr>
      <th>17</th>
      <td>99C-day-27__low</td>
      <td>depleted</td>
      <td>0.000418</td>
      <td>interpolated</td>
      <td>2392.117387</td>
    </tr>
    <tr>
      <th>18</th>
      <td>108C-day-18__low</td>
      <td>not depleted</td>
      <td>0.000065</td>
      <td>interpolated</td>
      <td>15482.242047</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108C-day-18__low</td>
      <td>depleted</td>
      <td>0.000487</td>
      <td>interpolated</td>
      <td>2052.442693</td>
    </tr>
    <tr>
      <th>20</th>
      <td>120C-day-10__low</td>
      <td>not depleted</td>
      <td>0.000029</td>
      <td>interpolated</td>
      <td>34145.783623</td>
    </tr>
    <tr>
      <th>21</th>
      <td>120C-day-10__low</td>
      <td>depleted</td>
      <td>0.000287</td>
      <td>interpolated</td>
      <td>3483.067791</td>
    </tr>
    <tr>
      <th>22</th>
      <td>180C-day-36__low</td>
      <td>not depleted</td>
      <td>0.000048</td>
      <td>interpolated</td>
      <td>20940.398173</td>
    </tr>
    <tr>
      <th>23</th>
      <td>180C-day-36__low</td>
      <td>depleted</td>
      <td>0.000323</td>
      <td>interpolated</td>
      <td>3100.452186</td>
    </tr>
    <tr>
      <th>24</th>
      <td>99C-day-27__medium</td>
      <td>not depleted</td>
      <td>0.000139</td>
      <td>interpolated</td>
      <td>7176.352127</td>
    </tr>
    <tr>
      <th>25</th>
      <td>99C-day-27__medium</td>
      <td>depleted</td>
      <td>0.000913</td>
      <td>interpolated</td>
      <td>1095.373755</td>
    </tr>
    <tr>
      <th>26</th>
      <td>108C-day-18__medium</td>
      <td>not depleted</td>
      <td>0.000069</td>
      <td>interpolated</td>
      <td>14551.386810</td>
    </tr>
    <tr>
      <th>27</th>
      <td>108C-day-18__medium</td>
      <td>depleted</td>
      <td>0.000792</td>
      <td>interpolated</td>
      <td>1262.226126</td>
    </tr>
    <tr>
      <th>28</th>
      <td>120C-day-10__medium</td>
      <td>not depleted</td>
      <td>0.000043</td>
      <td>interpolated</td>
      <td>23525.687102</td>
    </tr>
    <tr>
      <th>29</th>
      <td>120C-day-10__medium</td>
      <td>depleted</td>
      <td>0.000458</td>
      <td>interpolated</td>
      <td>2185.240092</td>
    </tr>
    <tr>
      <th>30</th>
      <td>180C-day-36__medium</td>
      <td>not depleted</td>
      <td>0.000050</td>
      <td>interpolated</td>
      <td>20046.024435</td>
    </tr>
    <tr>
      <th>31</th>
      <td>180C-day-36__medium</td>
      <td>depleted</td>
      <td>0.000478</td>
      <td>interpolated</td>
      <td>2091.586425</td>
    </tr>
    <tr>
      <th>32</th>
      <td>63C-day-10__high</td>
      <td>not depleted</td>
      <td>0.000102</td>
      <td>interpolated</td>
      <td>9830.591738</td>
    </tr>
    <tr>
      <th>33</th>
      <td>63C-day-10__high</td>
      <td>depleted</td>
      <td>0.012598</td>
      <td>interpolated</td>
      <td>79.377273</td>
    </tr>
    <tr>
      <th>34</th>
      <td>64C-day-15__high</td>
      <td>not depleted</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2656.009243</td>
    </tr>
    <tr>
      <th>35</th>
      <td>64C-day-15__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>99C-day-27__high</td>
      <td>not depleted</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2655.584701</td>
    </tr>
    <tr>
      <th>37</th>
      <td>99C-day-27__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>108C-day-18__high</td>
      <td>not depleted</td>
      <td>0.000208</td>
      <td>interpolated</td>
      <td>4814.580041</td>
    </tr>
    <tr>
      <th>39</th>
      <td>108C-day-18__high</td>
      <td>depleted</td>
      <td>0.010558</td>
      <td>interpolated</td>
      <td>94.715361</td>
    </tr>
    <tr>
      <th>40</th>
      <td>120C-day-10__high</td>
      <td>not depleted</td>
      <td>0.000156</td>
      <td>interpolated</td>
      <td>6429.843480</td>
    </tr>
    <tr>
      <th>41</th>
      <td>120C-day-10__high</td>
      <td>depleted</td>
      <td>0.009512</td>
      <td>interpolated</td>
      <td>105.133687</td>
    </tr>
    <tr>
      <th>42</th>
      <td>180C-day-36__high</td>
      <td>not depleted</td>
      <td>0.000196</td>
      <td>interpolated</td>
      <td>5093.387369</td>
    </tr>
    <tr>
      <th>43</th>
      <td>180C-day-36__high</td>
      <td>depleted</td>
      <td>0.013110</td>
      <td>interpolated</td>
      <td>76.275648</td>
    </tr>
  </tbody>
</table>
</div>




```python
fitparams['ic50_is_bound'] = fitparams['ic50_bound'].apply(lambda x: True if x!='interpolated' else False)

```


```python
fitparams[['sample', 'cells']] = fitparams['serum'].str.split('__', 1, expand=True)
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
      <th>RBD-targeting antibodies</th>
      <th>ic50</th>
      <th>ic50_bound</th>
      <th>NT50</th>
      <th>ic50_is_bound</th>
      <th>sample</th>
      <th>cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63C-day-10__very low</td>
      <td>not depleted</td>
      <td>0.000031</td>
      <td>interpolated</td>
      <td>31968.943613</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63C-day-10__very low</td>
      <td>depleted</td>
      <td>0.000454</td>
      <td>interpolated</td>
      <td>2200.799322</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64C-day-15__very low</td>
      <td>not depleted</td>
      <td>0.000064</td>
      <td>interpolated</td>
      <td>15594.993852</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64C-day-15__very low</td>
      <td>depleted</td>
      <td>0.000199</td>
      <td>interpolated</td>
      <td>5029.538042</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99C-day-27__very low</td>
      <td>not depleted</td>
      <td>0.000096</td>
      <td>interpolated</td>
      <td>10374.149491</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>99C-day-27__very low</td>
      <td>depleted</td>
      <td>0.000435</td>
      <td>interpolated</td>
      <td>2296.488086</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>108C-day-18__very low</td>
      <td>not depleted</td>
      <td>0.000038</td>
      <td>interpolated</td>
      <td>26336.105034</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>108C-day-18__very low</td>
      <td>depleted</td>
      <td>0.000225</td>
      <td>interpolated</td>
      <td>4450.299723</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120C-day-10__very low</td>
      <td>not depleted</td>
      <td>0.000027</td>
      <td>interpolated</td>
      <td>36527.531964</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>120C-day-10__very low</td>
      <td>depleted</td>
      <td>0.000218</td>
      <td>interpolated</td>
      <td>4588.546268</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>10</th>
      <td>180C-day-36__very low</td>
      <td>not depleted</td>
      <td>0.000048</td>
      <td>interpolated</td>
      <td>21040.240657</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>11</th>
      <td>180C-day-36__very low</td>
      <td>depleted</td>
      <td>0.000301</td>
      <td>interpolated</td>
      <td>3322.323929</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>very low</td>
    </tr>
    <tr>
      <th>12</th>
      <td>63C-day-10__low</td>
      <td>not depleted</td>
      <td>0.000040</td>
      <td>interpolated</td>
      <td>24852.475407</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>low</td>
    </tr>
    <tr>
      <th>13</th>
      <td>63C-day-10__low</td>
      <td>depleted</td>
      <td>0.000704</td>
      <td>interpolated</td>
      <td>1420.647038</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14</th>
      <td>64C-day-15__low</td>
      <td>not depleted</td>
      <td>0.000081</td>
      <td>interpolated</td>
      <td>12328.306795</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>low</td>
    </tr>
    <tr>
      <th>15</th>
      <td>64C-day-15__low</td>
      <td>depleted</td>
      <td>0.000262</td>
      <td>interpolated</td>
      <td>3823.459538</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>low</td>
    </tr>
    <tr>
      <th>16</th>
      <td>99C-day-27__low</td>
      <td>not depleted</td>
      <td>0.000110</td>
      <td>interpolated</td>
      <td>9071.489070</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>low</td>
    </tr>
    <tr>
      <th>17</th>
      <td>99C-day-27__low</td>
      <td>depleted</td>
      <td>0.000418</td>
      <td>interpolated</td>
      <td>2392.117387</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>low</td>
    </tr>
    <tr>
      <th>18</th>
      <td>108C-day-18__low</td>
      <td>not depleted</td>
      <td>0.000065</td>
      <td>interpolated</td>
      <td>15482.242047</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>low</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108C-day-18__low</td>
      <td>depleted</td>
      <td>0.000487</td>
      <td>interpolated</td>
      <td>2052.442693</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>low</td>
    </tr>
    <tr>
      <th>20</th>
      <td>120C-day-10__low</td>
      <td>not depleted</td>
      <td>0.000029</td>
      <td>interpolated</td>
      <td>34145.783623</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>low</td>
    </tr>
    <tr>
      <th>21</th>
      <td>120C-day-10__low</td>
      <td>depleted</td>
      <td>0.000287</td>
      <td>interpolated</td>
      <td>3483.067791</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>low</td>
    </tr>
    <tr>
      <th>22</th>
      <td>180C-day-36__low</td>
      <td>not depleted</td>
      <td>0.000048</td>
      <td>interpolated</td>
      <td>20940.398173</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>low</td>
    </tr>
    <tr>
      <th>23</th>
      <td>180C-day-36__low</td>
      <td>depleted</td>
      <td>0.000323</td>
      <td>interpolated</td>
      <td>3100.452186</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>low</td>
    </tr>
    <tr>
      <th>24</th>
      <td>99C-day-27__medium</td>
      <td>not depleted</td>
      <td>0.000139</td>
      <td>interpolated</td>
      <td>7176.352127</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>25</th>
      <td>99C-day-27__medium</td>
      <td>depleted</td>
      <td>0.000913</td>
      <td>interpolated</td>
      <td>1095.373755</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>26</th>
      <td>108C-day-18__medium</td>
      <td>not depleted</td>
      <td>0.000069</td>
      <td>interpolated</td>
      <td>14551.386810</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>27</th>
      <td>108C-day-18__medium</td>
      <td>depleted</td>
      <td>0.000792</td>
      <td>interpolated</td>
      <td>1262.226126</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>28</th>
      <td>120C-day-10__medium</td>
      <td>not depleted</td>
      <td>0.000043</td>
      <td>interpolated</td>
      <td>23525.687102</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>29</th>
      <td>120C-day-10__medium</td>
      <td>depleted</td>
      <td>0.000458</td>
      <td>interpolated</td>
      <td>2185.240092</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>30</th>
      <td>180C-day-36__medium</td>
      <td>not depleted</td>
      <td>0.000050</td>
      <td>interpolated</td>
      <td>20046.024435</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>31</th>
      <td>180C-day-36__medium</td>
      <td>depleted</td>
      <td>0.000478</td>
      <td>interpolated</td>
      <td>2091.586425</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>32</th>
      <td>63C-day-10__high</td>
      <td>not depleted</td>
      <td>0.000102</td>
      <td>interpolated</td>
      <td>9830.591738</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>high</td>
    </tr>
    <tr>
      <th>33</th>
      <td>63C-day-10__high</td>
      <td>depleted</td>
      <td>0.012598</td>
      <td>interpolated</td>
      <td>79.377273</td>
      <td>False</td>
      <td>63C-day-10</td>
      <td>high</td>
    </tr>
    <tr>
      <th>34</th>
      <td>64C-day-15__high</td>
      <td>not depleted</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2656.009243</td>
      <td>False</td>
      <td>64C-day-15</td>
      <td>high</td>
    </tr>
    <tr>
      <th>35</th>
      <td>64C-day-15__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
      <td>64C-day-15</td>
      <td>high</td>
    </tr>
    <tr>
      <th>36</th>
      <td>99C-day-27__high</td>
      <td>not depleted</td>
      <td>0.000377</td>
      <td>interpolated</td>
      <td>2655.584701</td>
      <td>False</td>
      <td>99C-day-27</td>
      <td>high</td>
    </tr>
    <tr>
      <th>37</th>
      <td>99C-day-27__high</td>
      <td>depleted</td>
      <td>0.040000</td>
      <td>lower</td>
      <td>25.000000</td>
      <td>True</td>
      <td>99C-day-27</td>
      <td>high</td>
    </tr>
    <tr>
      <th>38</th>
      <td>108C-day-18__high</td>
      <td>not depleted</td>
      <td>0.000208</td>
      <td>interpolated</td>
      <td>4814.580041</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>high</td>
    </tr>
    <tr>
      <th>39</th>
      <td>108C-day-18__high</td>
      <td>depleted</td>
      <td>0.010558</td>
      <td>interpolated</td>
      <td>94.715361</td>
      <td>False</td>
      <td>108C-day-18</td>
      <td>high</td>
    </tr>
    <tr>
      <th>40</th>
      <td>120C-day-10__high</td>
      <td>not depleted</td>
      <td>0.000156</td>
      <td>interpolated</td>
      <td>6429.843480</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>high</td>
    </tr>
    <tr>
      <th>41</th>
      <td>120C-day-10__high</td>
      <td>depleted</td>
      <td>0.009512</td>
      <td>interpolated</td>
      <td>105.133687</td>
      <td>False</td>
      <td>120C-day-10</td>
      <td>high</td>
    </tr>
    <tr>
      <th>42</th>
      <td>180C-day-36__high</td>
      <td>not depleted</td>
      <td>0.000196</td>
      <td>interpolated</td>
      <td>5093.387369</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>high</td>
    </tr>
    <tr>
      <th>43</th>
      <td>180C-day-36__high</td>
      <td>depleted</td>
      <td>0.013110</td>
      <td>interpolated</td>
      <td>76.275648</td>
      <td>False</td>
      <td>180C-day-36</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>




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

## Plot IC50 values


```python
IC50 = (ggplot(fitparams, aes(x='cells', y='ic50', colour='RBD-targeting antibodies', group = 'RBD-targeting antibodies')) +
              geom_point(size=3) +
        geom_line(alpha=1) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=12),
                   axis_text_x=element_text(size=12, angle= 45),
                   legend_text=element_text(size=12),
                   legend_title=element_text(size=12),
                   axis_title_x=element_text(size=12),
                   strip_text = element_text(size=12)
                  ) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='IC50') +
              xlab('ACE2 expression in target cells') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = IC50.draw()
IC50.save(f'./{resultsdir}/IC50.pdf')
```


    
![png](virus_neutralization_files/virus_neutralization_22_0.png)
    


## Plot NT50 values


```python
NT50 = (ggplot(fitparams, aes(x='cells', y='NT50', colour='RBD-targeting antibodies', group = 'RBD-targeting antibodies')) +
              geom_point(size=3) +
             geom_line(alpha=1) +
             theme(figure_size=(15,1*df['serum'].nunique()),
                   axis_text=element_text(size=12),
                   axis_text_x=element_text(size=12, angle= 45),
                   legend_text=element_text(size=12),
                   legend_title=element_text(size=12),
                   axis_title_x=element_text(size=12),
                   strip_text = element_text(size=12)
                  ) +
                geom_hline(yintercept=config['NT50_LOD'], 
                linetype='dotted', 
                size=1, 
                alpha=0.6, 
                color=CBPALETTE[7]) +
              facet_wrap('sample', ncol = 4)+
              scale_y_log10(name='NT50') +
              xlab('ACE2 expression in target cells') +
             scale_color_manual(values=CBPALETTE[1:])
                 )

_ = NT50.draw()
NT50.save(f'./{resultsdir}/NT50.pdf')
```


    
![png](virus_neutralization_files/virus_neutralization_24_0.png)
    


## Plot neut curves for all samples


```python
fig, axes = fits.plotSera(
                          xlabel='serum dilution',
                          ncol=6,
                          widthscale=4,
                          heightscale=4,
                          titlesize=25, labelsize=25, ticksize=15, legendfontsize=24, yticklocs=[0,0.5,1],
                          markersize=8, linewidth=2,
                          virus_to_color_marker={
                          'depleted': ('#56B4E9', 'o'),
                          'not depleted': ('#E69F00', 'o')},
                          legendtitle='RBD-targeting antibodies'
                         )
```


    
![png](virus_neutralization_files/virus_neutralization_26_0.png)
    



```python

```
