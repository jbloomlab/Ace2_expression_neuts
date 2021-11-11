# Calculate titers of RBD mutant spike-pseudotyped lentiviruses


```python
import os
import warnings

import math
import numpy as np 

from IPython.display import display, HTML
import matplotlib.pyplot as plt

from neutcurve.colorschemes import CBMARKERS, CBPALETTE

import pandas as pd
from plotnine import *

import yaml
```


```python
warnings.simplefilter('ignore')
```

Read config



```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Make output directory if needed


```python
resultsdir=config['resultsdir']
os.makedirs(resultsdir, exist_ok=True)
```


```python
pd.read_csv('data/virus_titer_RLU/ACE2clones_Wu_1_RLU.csv')
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
      <th>plasmid</th>
      <th>replicate</th>
      <th>virus</th>
      <th>dilution</th>
      <th>uL_virus</th>
      <th>RLU_per_well</th>
      <th>date</th>
      <th>cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.250000</td>
      <td>25.000000</td>
      <td>4085192</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.125000</td>
      <td>12.500000</td>
      <td>613486</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.062500</td>
      <td>6.250000</td>
      <td>128746</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.031250</td>
      <td>3.125000</td>
      <td>45646</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.015625</td>
      <td>1.562500</td>
      <td>24070</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
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
    </tr>
    <tr>
      <th>111</th>
      <td>2800</td>
      <td>rep2</td>
      <td>Wuhan1_614G</td>
      <td>0.003906</td>
      <td>0.390625</td>
      <td>6673</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
    </tr>
    <tr>
      <th>112</th>
      <td>2800</td>
      <td>rep2</td>
      <td>Wuhan1_614G</td>
      <td>0.001953</td>
      <td>0.195312</td>
      <td>5191</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
    </tr>
    <tr>
      <th>113</th>
      <td>2800</td>
      <td>rep2</td>
      <td>Wuhan1_614G</td>
      <td>0.000977</td>
      <td>0.097656</td>
      <td>2915</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2800</td>
      <td>rep2</td>
      <td>Wuhan1_614G</td>
      <td>0.000488</td>
      <td>0.048828</td>
      <td>1228</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2800</td>
      <td>rep2</td>
      <td>Wuhan1_614G</td>
      <td>0.000244</td>
      <td>0.024414</td>
      <td>261</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 8 columns</p>
</div>




```python
titers = pd.read_csv(config['virus_titers'])

titers = (titers
          .assign(RLUperuL=lambda x: x['RLU_per_well'] / x['uL_virus'],
                  date=lambda x: x['date'].astype(str)
                 )
         )

display(HTML(titers.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>plasmid</th>
      <th>replicate</th>
      <th>virus</th>
      <th>dilution</th>
      <th>uL_virus</th>
      <th>RLU_per_well</th>
      <th>date</th>
      <th>cells</th>
      <th>RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.250000</td>
      <td>25.0000</td>
      <td>4085192</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>163407.68</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.125000</td>
      <td>12.5000</td>
      <td>613486</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>49078.88</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.062500</td>
      <td>6.2500</td>
      <td>128746</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>20599.36</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.031250</td>
      <td>3.1250</td>
      <td>45646</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>14606.72</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wuhan1_614G</td>
      <td>0.015625</td>
      <td>1.5625</td>
      <td>24070</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>15404.80</td>
    </tr>
  </tbody>
</table>



```python
ncol=min(8, titers['virus'].nunique())
nrow=math.ceil(titers['virus'].nunique() / ncol)

p = (ggplot(titers.dropna()
            ) +
     aes('uL_virus', 'RLU_per_well', group='replicate') +
     geom_point(size=1.5) +
     geom_line() +
     facet_wrap('~virus+date+cells', ncol=ncol) +
     scale_y_log10(name='RLU per well') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(5 * ncol, 20 * nrow),
           )
     )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'RLU-vs-uL.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/RLU-vs-uL.pdf



    
![png](virus_titers_files/virus_titers_9_1.png)
    



```python
p = (ggplot(titers.dropna()
            ) +
     aes('uL_virus', 'RLUperuL', group='replicate') +
     geom_point(size=1.5) +
     geom_line() +
     facet_wrap('~virus+date+cells', ncol=ncol) +
     scale_y_log10(name='RLU per uL') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(4 * ncol, 10 * nrow),
           ) 
     )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'RLUperuL.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/RLUperuL.pdf



    
![png](virus_titers_files/virus_titers_10_1.png)
    


From visual inspection of the above plots, it appears that only the 6 highest dilutions (i.e., >1.5uL of virus per well) are reliable enough to calculate titers. 


```python
average_titers = (titers
                  .dropna() # missing values for some replicates
                  .query('uL_virus > 1') # drop lowest concentration of virus
                  .groupby(['virus', 'replicate', 'date', 'cells'])
                  .agg(mean_RLUperuL=pd.NamedAgg(column='RLUperuL', aggfunc=np.mean))
                  .reset_index()
                 )

display(HTML(average_titers.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>virus</th>
      <th>replicate</th>
      <th>date</th>
      <th>cells</th>
      <th>mean_RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wuhan1_614G</td>
      <td>rep1</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_A</td>
      <td>1771.266667</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>rep1</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_C</td>
      <td>6343.790000</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>rep1</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
      <td>37624.700000</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>rep1</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>52619.488000</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>rep1</td>
      <td>241021</td>
      <td>HEK293T_Ace2</td>
      <td>67143.120000</td>
    </tr>
  </tbody>
</table>



```python
p = (ggplot(average_titers, 
            aes(x='cells', y='mean_RLUperuL', color='replicate')
           ) +
     geom_point(size=2.5, alpha=0.5)+
     theme_classic() +
     theme(axis_text_x=element_text(angle=90, vjust=1, hjust=0.5),
           figure_size=(average_titers['virus'].nunique()*3,2),
           axis_title_x=element_blank(),
          ) +
     scale_y_log10(limits=[1,1.1e6]) +
     ylab('relative luciferase units\nper uL')+
     labs(title='pseudovirus entry titers') +
     scale_color_manual(values=CBPALETTE)
    )

_ = p.draw()

plotfile = os.path.join(resultsdir, 'virus_titers.pdf')
print(f"Saving to {plotfile}")
p.save(plotfile, verbose=False)
```

    Saving to results/virus_titers.pdf



    
![png](virus_titers_files/virus_titers_13_1.png)
    


Calculate how much virus to use in neut assays:


```python
target_RLU = 2e5
uL_virus_per_well = 50

dilute_virus = (average_titers
                .groupby(['virus', 'date', 'cells'])
                .agg(RLUperuL=pd.NamedAgg(column='mean_RLUperuL', aggfunc=np.mean))
                .reset_index()
                .assign(target_RLU = target_RLU,
                        uL_virus_per_well = uL_virus_per_well,
                        dilution_factor = lambda x: x['RLUperuL']/target_RLU*uL_virus_per_well,
                        uL_per_8mL = lambda x: 8000/x['dilution_factor'],
                        media_for_8ml = lambda x: 8000 - 8000/x['dilution_factor']
                       )
               )


titerfile = os.path.join(resultsdir, 'virus_titers.csv')
print(f"Saving to {titerfile}")

dilute_virus.to_csv(titerfile, index=False)

display(HTML(dilute_virus.head().to_html(index=False)))
```

    Saving to results/virus_titers.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>virus</th>
      <th>date</th>
      <th>cells</th>
      <th>RLUperuL</th>
      <th>target_RLU</th>
      <th>uL_virus_per_well</th>
      <th>dilution_factor</th>
      <th>uL_per_8mL</th>
      <th>media_for_8ml</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wuhan1_614G</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_A</td>
      <td>1830.921667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>0.457730</td>
      <td>17477.536359</td>
      <td>-9477.536359</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_C</td>
      <td>8123.145000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>2.030786</td>
      <td>3939.360925</td>
      <td>4060.639075</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>241021</td>
      <td>293T_ACE2_Clone_G</td>
      <td>39923.311667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>9.980828</td>
      <td>801.536713</td>
      <td>7198.463287</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>241021</td>
      <td>293T_consensus_kozak</td>
      <td>50877.348000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>12.719337</td>
      <td>628.963601</td>
      <td>7371.036399</td>
    </tr>
    <tr>
      <td>Wuhan1_614G</td>
      <td>241021</td>
      <td>HEK293T_Ace2</td>
      <td>64376.044000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>16.094011</td>
      <td>497.079317</td>
      <td>7502.920683</td>
    </tr>
  </tbody>
</table>



```python
# !jupyter nbconvert calculate_titer.ipynb --to HTML
```


```python

```
