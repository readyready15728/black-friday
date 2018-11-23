import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('cleaned.csv')

target = 'Purchase'
features = [c for c in df.columns if c != target]
categorical_features = [f for f in features if df.dtypes[f] == object]
numerical_features = [f for f in features if f not in categorical_features]

numerical_features = df[numerical_features].values

# Saving encoders makes it possible to reconstruct categorical variables from
# preprocessed data if desired (use method inverse_transform)
encoders = {}
encoded_features = []

for f in categorical_features:
    encoders[f] = OrdinalEncoder()
    encoded_features.append(encoders[f].fit_transform(df[f].values.reshape(-1, 1)))

categorical_features = np.column_stack(encoded_features)
features = np.concatenate([numerical_features, categorical_features], axis=1)

target = df[target]

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42)

if not Path('model.pkl').is_file():
    param_grid = {'colsample_bytree': [0.8, 1],
                  # gamma of 0 was shown to be quite useless in previous runs and has been omitted
                  'gamma': [1, 5],
                  'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                  'max_depth': [3, 5, 7],
                  # I am using a dual-core processor; change n_jobs to suit environment
                  'n_jobs': [2],
                  'subsample': [0.8, 1]}

    model = RandomizedSearchCV(xgb.XGBRegressor(), param_grid, cv=3, n_iter=100, verbose=100)
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

y_pred = model.predict(X_test)

print('SD of y_test:', '%.2f' % np.std(y_test))
print('RMSE of y_pred:', '%.2f' % np.sqrt(mean_squared_error(y_pred, y_test)))
