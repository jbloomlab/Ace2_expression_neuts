# Plot ACE2 expression vs Infectivity

This notebook plots pseudovirus infectivity in HEK293T clones expressing different levels of ACE2.


```python
import os
import warnings

import pandas as pd
from plotnine import *

import yaml
```


```python
warnings.simplefilter('ignore')
```


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```


```python
resultsdir=config['resultsdir']
os.makedirs(resultsdir, exist_ok=True)
```


```python
df = pd.read_csv(config['ACE2_expression_df'])
df
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
      <th>cells</th>
      <th>MFI (mean)</th>
      <th>RLU/ul</th>
      <th>relative MFI</th>
      <th>relative RLU/ul</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>49768</td>
      <td>307650.78400</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>medium</td>
      <td>5130</td>
      <td>160380.60100</td>
      <td>0.103078</td>
      <td>0.521307</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>2929</td>
      <td>105836.51500</td>
      <td>0.058853</td>
      <td>0.344015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>very low</td>
      <td>1481</td>
      <td>37566.13167</td>
      <td>0.029758</td>
      <td>0.122106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vero</td>
      <td>1340</td>
      <td>NaN</td>
      <td>0.026925</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_order = ['vero', 'very low', 'low', 'medium', 'high']
df['cells'] = pd.Categorical(df['cells'], categories=cat_order, ordered=True)
df
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
      <th>cells</th>
      <th>MFI (mean)</th>
      <th>RLU/ul</th>
      <th>relative MFI</th>
      <th>relative RLU/ul</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>49768</td>
      <td>307650.78400</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>medium</td>
      <td>5130</td>
      <td>160380.60100</td>
      <td>0.103078</td>
      <td>0.521307</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>2929</td>
      <td>105836.51500</td>
      <td>0.058853</td>
      <td>0.344015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>very low</td>
      <td>1481</td>
      <td>37566.13167</td>
      <td>0.029758</td>
      <td>0.122106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vero</td>
      <td>1340</td>
      <td>NaN</td>
      <td>0.026925</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
ACE2_infectivity=(
    ggplot(df) +
    aes(x="relative MFI", y="relative RLU/ul") +
    geom_point(size=2) +
    geom_text(
        mapping=aes(label='cells'),
        ha='left',
        adjust_text={'expand_points': (1.2, 1.2)},
        size=9
              ) +   
    theme_classic() +
    theme(figure_size=(3,3)) +
    labs(
        title = "Infectivity vs ACE2 Expression",
        x="Relative ACE2 expression",
        y="Relative infectivity") +
    scale_x_log10()
)

_ = ACE2_infectivity.draw()
```


    
![png](ACE2_expression_vs_infectivity_files/ACE2_expression_vs_infectivity_7_0.png)
    



```python
# version with Y axis as log scale
ACE2_infectivity_log=(
    ggplot(df)+
    aes(x="relative MFI", y="relative RLU/ul") +
    geom_point(size=2) +
    geom_text(
        mapping=aes(label='cells'),
        ha='left',
        adjust_text={'expand_points': (1.2, 1.2)},
        size=9
              ) +
    theme_classic()+
    theme(figure_size=(3,3)) +
    labs(
        #title = "Infectivity vs ACE2 Expression",
        x="Relative ACE2 expression",
        y="Relative infectivity"
         ) +
    scale_x_log10(breaks = (0.0625, 0.25, 1)) +
    scale_y_log10(breaks = (0.0625, 0.125, 0.25, 0.5, 1)) 
                      )

_ = ACE2_infectivity_log.draw()
```


    
![png](ACE2_expression_vs_infectivity_files/ACE2_expression_vs_infectivity_8_0.png)
    



```python
#ACE2 expression plot for figure 1
ACE2_expression=(
    ggplot(df)+
    aes(x="cells", y="MFI (mean)") +
    geom_point(size=2) +
    theme_classic()+
    theme(figure_size=(3,3),
          axis_text_x=element_text(size=10, angle=90)
          )+
    scale_y_log10(limits=[1,1.1e6])+
    xlab('Cell line') +
    scale_y_log10(limits=[1e2,1e5]) +
    labs(y ='Mean ACE2 expression')
)

_ = ACE2_expression.draw()
```


    
![png](ACE2_expression_vs_infectivity_files/ACE2_expression_vs_infectivity_9_0.png)
    



```python

```
