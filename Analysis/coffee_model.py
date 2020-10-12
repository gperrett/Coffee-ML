import sklearn
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



df = pd.read_csv('/Users/georgeperrett/Git_Proj/Coffee-ML/Data/merged_data_cleaned.csv')

features = ['Cupper.Points',
'Aroma', 'Flavor', 'Aftertaste',
            'Acidity', 'Body', 'Balance',
            'Uniformity', 'Clean.Cup', 'Sweetness',
            'Country.of.Origin', 'Processing.Method']


def clean(df):
    df = df.loc[:, features]

    # recode non common Country of Origins as others
    c = pd.DataFrame(df['Country.of.Origin'].value_counts() < 20)
    c = c[c['Country.of.Origin'] == True]
    c = list(c.index)
    df['Country.of.Origin'] = df['Country.of.Origin'].replace(c, 'Other')
    df = df[(df != 0).all(1)]
    return(df)

df= clean(df)



# seperate fimal hold out set
train, test = train_test_split(df, test_size = .2, random_state = 1234)
y_train = train['Cupper.Points']
train = train.drop(['Cupper.Points'], axis = 1)
y_test = test['Cupper.Points']
test = test.drop(['Cupper.Points'], axis = 1)



# build data pipeline
cat_attributes = train.select_dtypes(exclude = ['int', 'float']).columns
num_attributes = train.select_dtypes(include = ['int', 'float']).columns

num_pipeline = sklearn.pipeline.Pipeline([
('imputer', SimpleImputer(strategy='median')),
('std_scale', sklearn.preprocessing.StandardScaler()),
])

cat_piplein = sklearn.pipeline.Pipeline([
('imputer', SimpleImputer(strategy='most_frequent')),
('one_hot', sklearn.preprocessing.OneHotEncoder()),
])

full_pipeline = ColumnTransformer([
('num', num_pipeline, num_attributes),
('cat', cat_piplein, cat_attributes),
])

train = full_pipeline.fit_transform(train)
test = full_pipeline.transform(test)

names = list(num_attributes) + list(full_pipeline.named_transformers_['cat']['one_hot'].get_feature_names())

reg = LassoCV(cv = 5,n_alphas = 10000, n_jobs = -1, random_state = 1234).fit(train, y_train)
coefficents = pd.DataFrame(reg.coef_, names)
selected = coefficents[coefficents[0]!=0]
selected.columns = ['Weight']

preds = reg.predict(test)
rmse = mean_squared_error(preds, y_test)


lasso = Lasso(max_iter = 10000)
coefs = []

for a in reg.alphas_:
    lasso.set_params(alpha=a)
    lasso.fit(train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(reg.alphas_, coefs)
ax.set_xscale('log')
plt.vlines(x = reg.alpha_,ymin=-0.15, ymax=0.15, ls = '--')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('weights')
