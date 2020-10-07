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


select = pd.concat([df[['Country.of.Origin', 'Processing.Method','Total.Cup.Points']], select], axis = 1)

plot_df = pd.melt(select,id_vars = list(select.columns)[0:3] , value_vars= list(select.columns)[3:12])

plot_df = plot_df[(plot_df['Total.Cup.Points'] > 0)& (plot_df['value'] > 0)]

g = sns.FacetGrid(plot_df, col = 'variable', col_wrap=3).add_legend()
g.map_dataframe(sns.regplot,x= 'value', y = 'Total.Cup.Points', x_jitter=.2, y_jitter=.1)
g.add_legend()


sns.kdeplot(df['Total.Cup.Points'],hue = df['Country.of.Origin'], shade = True,legend = 2)
sns.violinplot(x = "variable", y = "value", data = plot_df)
