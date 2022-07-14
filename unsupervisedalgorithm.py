import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly
import json



def clean_data():
  # Read CSV
  sales_df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
  # Drop Columns
  sales_df = sales_df.drop(['ORDERDATE', 'STATUS', 'QTR_ID', 'ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER'],axis=1)
  # Add dummies
  for i in ['COUNTRY', 'PRODUCTLINE', 'DEALSIZE']:
    dummy = pd.get_dummies(sales_df[i])
    sales_df.drop(columns = i, inplace=True)
    sales_df = pd.concat([sales_df, dummy], axis=1)
  # Add categorical
  sales_df["PRODUCTCODE"] = pd.Categorical(sales_df["PRODUCTCODE"]).codes
  return sales_df




# KMean Clustring
def kmean():
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(clean_data())
    # Scaled Data
    kmeans = KMeans(5)
    kmeans.fit(sales_df_scaled)
    kmeans.fit_predict(sales_df_scaled)

    # PCA Component
    pca = PCA(n_components = 3)
    principal_comp = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(data=principal_comp, columns=['pca1','pca2','pca3'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':kmeans.labels_})], axis=1)
    fig = px.scatter_3d(pca_df, x='pca1',y='pca2', z='pca3', color='cluster', symbol='cluster', opacity=0.7)
    fig.update_layout(margin = dict(l=0, r=0, b=0, t=0))
    graph3JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph3JSON






    