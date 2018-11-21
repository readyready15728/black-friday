import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
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
    # Parameters suggested by previous top scorers from a RandomizedSearchCV
    param_grid = {'colsample_bytree': [1],
                  'learning_rate': [0.05, 0.1, 0.3],
                  'max_depth': [5, 8],
                  'n_estimators': [100]}

    model = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=3, verbose=100)
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

y_pred = model.predict(X_test)

print('SD of y_test:', np.std(y_test))
print('RMSE of y_pred:', np.sqrt(mean_squared_error(y_pred, y_test)))
