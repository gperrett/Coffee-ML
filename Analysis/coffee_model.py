import sklearn
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV




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

test_pct = math.floor((len(df)*.1))
val_pct = math.floor((len(df)*.2))

# seperate fimal hold out set
train, test = train_test_split(df, test_size = test_pct, random_state = 2)
y_train = train['Cupper.Points']
train = train.drop(['Cupper.Points'], axis = 1)
y_test = test['Cupper.Points']
test = test.drop(['Cupper.Points'], axis = 1)

def lasso(train, test):

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

    reg = LassoCV(cv = 10, n_jobs = -1, random_state = 2).fit(train, y_train)
    coefficents = pd.DataFrame(reg.coef_, names)
    preds = reg.predict(test)
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(preds, y_test)

    results = {'coefficents': coefficents, 'rmse': error, 'train': train, 'test':test, 'y_test': y_test, 'y_train': y_train}
    return(results)

lasso_fit = lasso(train = train, test = test)
