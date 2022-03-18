# Calculate titers of spike-pseudotyped lentiviruses on different cell lines


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
      <th>2727_plasmid</th>
      <th>amphoB</th>
      <th>RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wu_1</td>
      <td>0.50000</td>
      <td>50.000</td>
      <td>7380696</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>147613.92</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wu_1</td>
      <td>0.25000</td>
      <td>25.000</td>
      <td>1875214</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>75008.56</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wu_1</td>
      <td>0.12500</td>
      <td>12.500</td>
      <td>385228</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>30818.24</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wu_1</td>
      <td>0.06250</td>
      <td>6.250</td>
      <td>91940</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>14710.40</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>rep1</td>
      <td>Wu_1</td>
      <td>0.03125</td>
      <td>3.125</td>
      <td>27958</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>8946.56</td>
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
     facet_wrap('~virus+date+cells+2727_plasmid+amphoB', ncol=ncol) +
     scale_y_log10(name='RLU per well') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(5 * ncol, 40 * nrow),
           )
     )

_ = p.draw()
```


    
![png](virus_titers_files/virus_titers_8_0.png)
    



```python
p = (ggplot(titers.dropna()
            ) +
     aes('uL_virus', 'RLUperuL', group='replicate') +
     geom_point(size=1.5) +
     geom_line() +
     facet_wrap('~virus+date+cells+2727_plasmid+amphoB', ncol=ncol) +
     scale_y_log10(name='RLU per uL') +
     scale_x_log10(name='uL virus per well') +
     theme_classic() +
     theme(axis_text_x=element_text(angle=90),
           figure_size=(4 * ncol, 40 * nrow),
           ) 
     )

_ = p.draw()
```


    
![png](virus_titers_files/virus_titers_9_0.png)
    


From visual inspection of the above plots, it appears that only the 5 highest dilutions (i.e., >1uL of virus per well) are reliable enough to calculate titers. 


```python
average_titers = (titers
                  .dropna() # missing values for some replicates
                  .query('uL_virus > 1') # drop lowest concentration of virus
                  .groupby(['virus', 'replicate', 'date', 'cells', '2727_plasmid', 'amphoB'])
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
      <th>2727_plasmid</th>
      <th>amphoB</th>
      <th>mean_RLUperuL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wu_1</td>
      <td>rep1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Bernadeta_stock</td>
      <td>amphoB</td>
      <td>48910.933333</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>rep1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>5959.903333</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>rep1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_60hpt</td>
      <td>amphoB</td>
      <td>47823.730000</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>rep1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_60hpt</td>
      <td>no_amphoB</td>
      <td>5281.416667</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>rep1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2021_60hpt</td>
      <td>amphoB</td>
      <td>70423.453333</td>
    </tr>
  </tbody>
</table>



```python
p = (ggplot(average_titers, 
            aes(x='cells', y='mean_RLUperuL', color='date', shape = 'amphoB')
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
```


    
![png](virus_titers_files/virus_titers_12_0.png)
    


Calculate how much virus to use in neut assays:


```python
target_RLU = 2e5
uL_virus_per_well = 50

dilute_virus = (average_titers
                .groupby(['virus', 'date', 'cells', '2727_plasmid','amphoB'])
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

display(HTML(dilute_virus.to_html(index=False)))
```

    Saving to results/virus_titers.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>virus</th>
      <th>date</th>
      <th>cells</th>
      <th>2727_plasmid</th>
      <th>amphoB</th>
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
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Bernadeta_stock</td>
      <td>amphoB</td>
      <td>48134.986667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>12.033747</td>
      <td>664.797109</td>
      <td>7335.202891</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>6035.725000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>1.508931</td>
      <td>5301.765737</td>
      <td>2698.234263</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_60hpt</td>
      <td>amphoB</td>
      <td>47475.220000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>11.868805</td>
      <td>674.035844</td>
      <td>7325.964156</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2020_60hpt</td>
      <td>no_amphoB</td>
      <td>5345.510000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>1.336378</td>
      <td>5986.332455</td>
      <td>2013.667545</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2021_60hpt</td>
      <td>amphoB</td>
      <td>71130.041667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>17.782510</td>
      <td>449.880237</td>
      <td>7550.119763</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Clone_A</td>
      <td>Maxi2021_60hpt</td>
      <td>no_amphoB</td>
      <td>7591.453333</td>
      <td>200000.0</td>
      <td>50</td>
      <td>1.897863</td>
      <td>4215.266642</td>
      <td>3784.733358</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_48hpt</td>
      <td>no_amphoB</td>
      <td>49419.021667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>12.354755</td>
      <td>647.523948</td>
      <td>7352.476052</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_60hpt</td>
      <td>amphoB</td>
      <td>153364.706667</td>
      <td>200000.0</td>
      <td>50</td>
      <td>38.341177</td>
      <td>208.652960</td>
      <td>7791.347040</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2020_60hpt</td>
      <td>no_amphoB</td>
      <td>41463.535000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>10.365884</td>
      <td>771.762466</td>
      <td>7228.237534</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2021_60hpt</td>
      <td>amphoB</td>
      <td>228904.052000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>57.226013</td>
      <td>139.796564</td>
      <td>7860.203436</td>
    </tr>
    <tr>
      <td>Wu_1</td>
      <td>200122</td>
      <td>293T_ACE2_Consensus_Kozak</td>
      <td>Maxi2021_60hpt</td>
      <td>no_amphoB</td>
      <td>56560.515000</td>
      <td>200000.0</td>
      <td>50</td>
      <td>14.140129</td>
      <td>565.765711</td>
      <td>7434.234289</td>
    </tr>
  </tbody>
</table>



```python
# !jupyter nbconvert calculate_titer.ipynb --to HTML
```


```python

```
