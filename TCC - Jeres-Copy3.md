## 0. Import das bibliotecas utilizadas 


```python
import pandas as pd
from prophet import Prophet
```


```python
import warnings; 
warnings.simplefilter('ignore')
```

## 1. Leitura do arquivo txt contendo as informações que compõem o cálculo da margem


```python
 df = pd.read_table("C:\\MBA\\TCC\\dataset_novo.txt", dtype ={'Cod_UN': 'str', 'SAFRA': 'str', 'Cod_ITEM': 'str','Cod_ESTABELECIMENTO': 'str', 'Cod_REGIONAL': 'str', 'Cod_CLIENTE': 'str', 'Cod_REPRESENTANTE': 'str', 'GRUPO_NEGOCIO': 'str', 'NUMERADOR': 'float', 'DENOMINADOR': 'float'}, sep=';');
```


```python
df.head()
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
      <th>Cod_UN</th>
      <th>Cod_ITEM</th>
      <th>SAFRA</th>
      <th>Cod_REGIONAL</th>
      <th>Cod_ESTABELECIMENTO</th>
      <th>DATA</th>
      <th>Cod_CLIENTE</th>
      <th>Cod_REPRESENTANTE</th>
      <th>GRUPO_NEGOCIO</th>
      <th>NUMERADOR</th>
      <th>DENOMINADOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001</td>
      <td>0010061</td>
      <td>SEM SAFRA</td>
      <td>073</td>
      <td>010</td>
      <td>01/05/2010</td>
      <td>64791</td>
      <td>10012</td>
      <td>BENS FORNECIMENTO</td>
      <td>97.34000</td>
      <td>394.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001</td>
      <td>0013108</td>
      <td>2019/2019</td>
      <td>016</td>
      <td>035</td>
      <td>01/12/2018</td>
      <td>660223</td>
      <td>15006</td>
      <td>BENS FORNECIMENTO</td>
      <td>22776.20000</td>
      <td>159457.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001</td>
      <td>0012638</td>
      <td>2017/2018</td>
      <td>068</td>
      <td>068</td>
      <td>01/09/2017</td>
      <td>9442</td>
      <td>14009</td>
      <td>BENS FORNECIMENTO</td>
      <td>249.05680</td>
      <td>2352.63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001</td>
      <td>0011580</td>
      <td>SEM SAFRA</td>
      <td>030</td>
      <td>030</td>
      <td>01/09/2014</td>
      <td>2113</td>
      <td>9013</td>
      <td>BENS FORNECIMENTO</td>
      <td>168.23000</td>
      <td>592.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001</td>
      <td>0013096</td>
      <td>2019/2020</td>
      <td>068</td>
      <td>014</td>
      <td>01/10/2019</td>
      <td>2143</td>
      <td>14006</td>
      <td>BENS FORNECIMENTO</td>
      <td>3620.36867</td>
      <td>14332.90</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Transformação dos dados

### 2.1 Inclusão de coluna no formato de data em razão da coluna existente estar no formato object


```python
df.dtypes
```




    Cod_UN                  object
    Cod_ITEM                object
    SAFRA                   object
    Cod_REGIONAL            object
    Cod_ESTABELECIMENTO     object
    DATA                    object
    Cod_CLIENTE             object
    Cod_REPRESENTANTE       object
    GRUPO_NEGOCIO           object
    NUMERADOR              float64
    DENOMINADOR            float64
    dtype: object




```python
df['ANO'] = df['DATA'].apply(lambda x: str(x)[-4:])
df['MES'] = df['DATA'].apply(lambda x: str(x)[3:5])
df['DIA'] = df['DATA'].apply(lambda x: str(x)[0:2])
df['ds'] = pd.DatetimeIndex(df['ANO']+'-'+df['MES']+'-'+df['DIA'])
```


```python
df.head()
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
      <th>Cod_UN</th>
      <th>Cod_ITEM</th>
      <th>SAFRA</th>
      <th>Cod_REGIONAL</th>
      <th>Cod_ESTABELECIMENTO</th>
      <th>DATA</th>
      <th>Cod_CLIENTE</th>
      <th>Cod_REPRESENTANTE</th>
      <th>GRUPO_NEGOCIO</th>
      <th>NUMERADOR</th>
      <th>DENOMINADOR</th>
      <th>ANO</th>
      <th>MES</th>
      <th>DIA</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001</td>
      <td>0010061</td>
      <td>SEM SAFRA</td>
      <td>073</td>
      <td>010</td>
      <td>01/05/2010</td>
      <td>64791</td>
      <td>10012</td>
      <td>BENS FORNECIMENTO</td>
      <td>97.34000</td>
      <td>394.00</td>
      <td>2010</td>
      <td>05</td>
      <td>01</td>
      <td>2010-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001</td>
      <td>0013108</td>
      <td>2019/2019</td>
      <td>016</td>
      <td>035</td>
      <td>01/12/2018</td>
      <td>660223</td>
      <td>15006</td>
      <td>BENS FORNECIMENTO</td>
      <td>22776.20000</td>
      <td>159457.58</td>
      <td>2018</td>
      <td>12</td>
      <td>01</td>
      <td>2018-12-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001</td>
      <td>0012638</td>
      <td>2017/2018</td>
      <td>068</td>
      <td>068</td>
      <td>01/09/2017</td>
      <td>9442</td>
      <td>14009</td>
      <td>BENS FORNECIMENTO</td>
      <td>249.05680</td>
      <td>2352.63</td>
      <td>2017</td>
      <td>09</td>
      <td>01</td>
      <td>2017-09-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001</td>
      <td>0011580</td>
      <td>SEM SAFRA</td>
      <td>030</td>
      <td>030</td>
      <td>01/09/2014</td>
      <td>2113</td>
      <td>9013</td>
      <td>BENS FORNECIMENTO</td>
      <td>168.23000</td>
      <td>592.00</td>
      <td>2014</td>
      <td>09</td>
      <td>01</td>
      <td>2014-09-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001</td>
      <td>0013096</td>
      <td>2019/2020</td>
      <td>068</td>
      <td>014</td>
      <td>01/10/2019</td>
      <td>2143</td>
      <td>14006</td>
      <td>BENS FORNECIMENTO</td>
      <td>3620.36867</td>
      <td>14332.90</td>
      <td>2019</td>
      <td>10</td>
      <td>01</td>
      <td>2019-10-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Cod_UN                         object
    Cod_ITEM                       object
    SAFRA                          object
    Cod_REGIONAL                   object
    Cod_ESTABELECIMENTO            object
    DATA                           object
    Cod_CLIENTE                    object
    Cod_REPRESENTANTE              object
    GRUPO_NEGOCIO                  object
    NUMERADOR                     float64
    DENOMINADOR                   float64
    ANO                            object
    MES                            object
    DIA                            object
    ds                     datetime64[ns]
    dtype: object



### 2.2 Exploração de Resultados Preditos para Bens de Fornecimento


```python
df_bens = df.loc[df['GRUPO_NEGOCIO']=='BENS FORNECIMENTO'].copy()
```


```python
## Agregação das informação que compõem o calculo de margem a nível de grupo de negócio
df_bens_gr = df_bens.groupby(['GRUPO_NEGOCIO', 'ds']).agg({'NUMERADOR':'sum','DENOMINADOR':'sum'}).reset_index()
```


```python
## Criação do campo target, no caso, margem de contribuição
df_bens_gr['y'] = df_bens_gr['NUMERADOR'] / df_bens_gr['DENOMINADOR']
```


```python
df_bens_gr.head()
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
      <th>GRUPO_NEGOCIO</th>
      <th>ds</th>
      <th>NUMERADOR</th>
      <th>DENOMINADOR</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-09-01</td>
      <td>3.331420e+06</td>
      <td>20127857.16</td>
      <td>0.165513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-10-01</td>
      <td>9.561512e+06</td>
      <td>51551921.46</td>
      <td>0.185473</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-11-01</td>
      <td>5.462440e+06</td>
      <td>27811348.11</td>
      <td>0.196410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-12-01</td>
      <td>3.160686e+06</td>
      <td>16961351.96</td>
      <td>0.186346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENS FORNECIMENTO</td>
      <td>2010-01-01</td>
      <td>8.984557e+05</td>
      <td>17572774.21</td>
      <td>0.051128</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_bens_gr.drop(['GRUPO_NEGOCIO', 'NUMERADOR', 'DENOMINADOR'], axis=1, inplace=True, errors='ignore')
```


```python
## Início da chamada do modelo preditivo
m = Prophet(interval_width=0.95, daily_seasonality=False, yearly_seasonality=True)
model = m.fit(df_bens_gr)
    
## Uso do modelo criado
future = m.make_future_dataframe(periods=12,freq='M')
forecast = m.predict(future)
    
forecast.head()
```

    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    




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
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-09-01</td>
      <td>0.227946</td>
      <td>0.126762</td>
      <td>0.308093</td>
      <td>0.227946</td>
      <td>0.227946</td>
      <td>-0.015951</td>
      <td>-0.015951</td>
      <td>-0.015951</td>
      <td>-0.015951</td>
      <td>-0.015951</td>
      <td>-0.015951</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.211995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-10-01</td>
      <td>0.227468</td>
      <td>0.122996</td>
      <td>0.307422</td>
      <td>0.227468</td>
      <td>0.227468</td>
      <td>-0.007343</td>
      <td>-0.007343</td>
      <td>-0.007343</td>
      <td>-0.007343</td>
      <td>-0.007343</td>
      <td>-0.007343</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.220126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-11-01</td>
      <td>0.226975</td>
      <td>0.135542</td>
      <td>0.324302</td>
      <td>0.226975</td>
      <td>0.226975</td>
      <td>0.008333</td>
      <td>0.008333</td>
      <td>0.008333</td>
      <td>0.008333</td>
      <td>0.008333</td>
      <td>0.008333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.235308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-12-01</td>
      <td>0.226497</td>
      <td>0.140145</td>
      <td>0.328968</td>
      <td>0.226497</td>
      <td>0.226497</td>
      <td>0.008627</td>
      <td>0.008627</td>
      <td>0.008627</td>
      <td>0.008627</td>
      <td>0.008627</td>
      <td>0.008627</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.235124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-01-01</td>
      <td>0.226004</td>
      <td>0.126840</td>
      <td>0.303804</td>
      <td>0.226004</td>
      <td>0.226004</td>
      <td>-0.011394</td>
      <td>-0.011394</td>
      <td>-0.011394</td>
      <td>-0.011394</td>
      <td>-0.011394</td>
      <td>-0.011394</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.214610</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_df_gn = m.plot(forecast)
```


    
![png](output_19_0.png)
    



```python
m.plot_components(forecast)
```




    
![png](output_20_0.png)
    




    
![png](output_20_1.png)
    


## 3. Tratamento para Modelo visando a predição do Grupo de Negócio


```python
## Agregação das informação que compõem o calculo de margem a nível de grupo de negócio
df_gn = df.groupby(['GRUPO_NEGOCIO', 'ds']).agg({'NUMERADOR':'sum','DENOMINADOR':'sum'}).reset_index()
```


```python
## Criação do campo target, no caso, margem de contribuição
df_gn['y'] = df_gn['NUMERADOR'] / df_gn['DENOMINADOR']
```


```python
df_gn.head()
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
      <th>GRUPO_NEGOCIO</th>
      <th>ds</th>
      <th>NUMERADOR</th>
      <th>DENOMINADOR</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-09-01</td>
      <td>3.331420e+06</td>
      <td>20127857.16</td>
      <td>0.165513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-10-01</td>
      <td>9.561512e+06</td>
      <td>51551921.46</td>
      <td>0.185473</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-11-01</td>
      <td>5.462440e+06</td>
      <td>27811348.11</td>
      <td>0.196410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENS FORNECIMENTO</td>
      <td>2009-12-01</td>
      <td>3.160686e+06</td>
      <td>16961351.96</td>
      <td>0.186346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENS FORNECIMENTO</td>
      <td>2010-01-01</td>
      <td>8.984557e+05</td>
      <td>17572774.21</td>
      <td>0.051128</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_gn.drop(['NUMERADOR', 'DENOMINADOR'], axis=1, inplace=True, errors='ignore')
```

### 3.1 Implementação do Prophet


```python
for i in df_gn['GRUPO_NEGOCIO'].unique():
    
    ## Guarda as informações do dataset do grupo utilizado no loop
    df_gn_aux = df_gn.loc[df_gn['GRUPO_NEGOCIO'] == i].copy()
    
    ## reduz as informações somente para as colunas ds e y
    df_gn_hist = df_gn.loc[df_gn['GRUPO_NEGOCIO'] == i].copy()
    df_gn_aux2 = df_gn_hist.copy()
    df_gn_aux2.drop(['GRUPO_NEGOCIO'], axis=1, inplace=True, errors='ignore')
    
    ## Início da chamada do modelo preditivo
    m = Prophet(interval_width=0.95, daily_seasonality=False, yearly_seasonality=True)
    model = m.fit(df_gn_aux2)
    
    ## Uso do modelo criado
    future = m.make_future_dataframe(periods=12,freq='M')
    forecast = m.predict(future)
   
    ## merge dos valores preditos com o dataset armanezado (df_gn_aux) e cria um arquivo com a respectiva categoria
    df_gn_aux = pd.merge(df_gn_hist, forecast)
    df_gn_aux.to_csv(r'C:\\MBA\\TCC\\CSV\\pred_grupo_' + i + '.csv', index = False, sep = ';')

```

    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    

## 4.1 Tratamento para Modelo visando a predição por estabelecimento


```python
## Agregação das informação que compõem o calculo de margem a nível de estabelecimento e grupo de negócio
df_estab_gn = df.groupby(['Cod_ESTABELECIMENTO','GRUPO_NEGOCIO', 'ds']).agg({'NUMERADOR':'sum','DENOMINADOR':'sum'}).reset_index()
```


```python
## Remove nulos no campo Cod_ESTABELECIMENTO ou GRUPO_NEGOCIO
df_estab_gn = df_estab_gn[df_estab_gn['Cod_ESTABELECIMENTO'].notna()]
```


```python
## Criação do campo target, no caso, margem de contribuição
df_estab_gn['y'] = df_estab_gn['NUMERADOR'] / df_estab_gn['DENOMINADOR']
```


```python
df_estab_gn.head()
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
      <th>Cod_ESTABELECIMENTO</th>
      <th>GRUPO_NEGOCIO</th>
      <th>ds</th>
      <th>NUMERADOR</th>
      <th>DENOMINADOR</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001</td>
      <td>BENS FORNECIMENTO</td>
      <td>2009-11-01</td>
      <td>2250.000</td>
      <td>14500.0</td>
      <td>0.155172</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001</td>
      <td>BENS FORNECIMENTO</td>
      <td>2009-12-01</td>
      <td>-77460.955</td>
      <td>312181.0</td>
      <td>-0.248128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001</td>
      <td>BENS FORNECIMENTO</td>
      <td>2010-01-01</td>
      <td>-1773.070</td>
      <td>68100.0</td>
      <td>-0.026036</td>
    </tr>
    <tr>
      <th>3</th>
      <td>001</td>
      <td>BENS FORNECIMENTO</td>
      <td>2010-02-01</td>
      <td>2198.800</td>
      <td>10244.8</td>
      <td>0.214626</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001</td>
      <td>BENS FORNECIMENTO</td>
      <td>2010-03-01</td>
      <td>-371098.430</td>
      <td>26315.5</td>
      <td>-14.101895</td>
    </tr>
  </tbody>
</table>
</div>



### 4.1.2 Implementação do Prophet


```python
for i in df_estab_gn['GRUPO_NEGOCIO'].unique():
    ## restringe dataset somente ao grupo de negócio da iteração
    df_estab_gn_aux = df_estab_gn.loc[df_estab_gn['GRUPO_NEGOCIO'] == i].copy()
    ## novo loop para os estabelecimentos
    for j in df_estab_gn_aux['Cod_ESTABELECIMENTO'].unique():
        ## dataset contemplando o estabelecimento utilizado na iteração
        df_estab_gn_hist = df_estab_gn_aux.loc[df_estab_gn_aux['Cod_ESTABELECIMENTO'] == j].copy()
        df_estab_gn_aux2 = df_estab_gn_hist.copy()
        ## dataset somente com as colunas utilizadas para a predição
        df_estab_gn_aux2.drop(['GRUPO_NEGOCIO', 'Cod_ESTABELECIMENTO', 'NUMERADOR', 'DENOMINADOR'], axis=1, inplace = True, errors = 'ignore')
        
        ## início da chamada do Prophet
        m = Prophet(interval_width=0.95, daily_seasonality=False, yearly_seasonality=True)
        model = m.fit(df_estab_gn_aux2)
        
        ## aplicação do modelo criado
        future = m.make_future_dataframe(12, freq='M')
        forecast = m.predict(future)
        
        ## merge dos valores preditos com o dataset armanezado (df_estab_gn_aux2) e cria um arquivo com a respectiva categoria
        df_estab_gn_final = pd.merge(df_estab_gn_hist, forecast)
        df_estab_gn_final.to_csv(r'C:\\MBA\\TCC\\CSV\\Estab\\pred_estab_' + j + '_grp_' + i + '.csv', index = False, sep = ';')
```

    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 7.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 0.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 19.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 17.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 20.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 14.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 17.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 21.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 7.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 1.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 1.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 24.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 0.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 22.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 18.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 7.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 8.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 0.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 5.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 18.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 20.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 9.
    INFO:cmdstanpy:start chain 1
    INFO:cmdstanpy:finish chain 1
    


```python

```
