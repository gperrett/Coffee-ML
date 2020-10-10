import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv('/Users/georgeperrett/Git_Proj/Coffee-ML/Data/merged_data_cleaned.csv')


# recode infreuent conutires to other
c = pd.DataFrame(df['Country.of.Origin'].value_counts() < 20)
c = c[c['Country.of.Origin'] == True]
c = list(c.index)
df['Country.of.Origin'] = df['Country.of.Origin'].replace(c, 'Other')


select = df.loc[:,'Aroma':'Sweetness']

corr = select.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .3})


pca = PCA()
reduced = pca.fit_transform(select)

reduced_df = pd.DataFrame(reduced)


select = pd.concat([df[['Country.of.Origin', 'Processing.Method','Cupper.Points']], select], axis = 1)

plot_df = pd.melt(select,id_vars = list(select.columns)[0:3] , value_vars= list(select.columns)[3:12])

plot_df = plot_df[(plot_df['Cupper.Points'] > 0)& (plot_df['value'] > 0)]

g = sns.FacetGrid(plot_df, col = 'variable', col_wrap=3).add_legend()
g.map_dataframe(sns.regplot,x= 'value', y = 'Cupper.Points', x_jitter=.2, y_jitter=.1)
g.add_legend()


sns.kdeplot(df['Cupper.Points'],hue = df['Country.of.Origin'])

g = sns.FacetGrid(plot_df, col = 'Country.of.Origin', col_wrap = 4).add_legend()
g.map_dataframe(sns.kdeplot, x = 'Cupper.Points')
g.add_legend()

sns.boxplot(x = "Cupper.Points", y = "Country.of.Origin", data = plot_df)
sns.boxplot(x = "Cupper.Points", y = "Country.of.Origin", data = plot_df)


sns.violinplot(y = 'Country.of.Origin', x = "Cupper.Points", data = df)
sns.violinplot(x = "variable", y = "value", data = plot_df)
