import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error



#RandomForestRegressor

# Funktion zur Erstellung von X und y
def create_train_X_y(df):
    train_X = train_df.iloc[:-1].values.reshape(-1, 1)
    train_y = train_df.iloc[1:].values.reshape(-1, 1)
    return train_X, train_y

def create_test_X_y(df):
    test_X = test_df.iloc[:-1].values.reshape(-1, 1)
    test_y = test_df.iloc[1:].values.reshape(-1, 1)
    return test_X, test_y

X_train, y_train = create_train_X_y(train_df)
X_test, y_test = create_test_X_y(test_df)

model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],  # Anzahl der Bäume
    'max_depth': [None, 10, 20, 30],  # Maximale Tiefe der Bäume
    'min_samples_split': [2, 5, 10],  # Minimale Anzahl von Proben, um einen internen Knoten zu teilen
    'min_samples_leaf': [1, 2, 4],    # Minimale Anzahl von Proben in einem Blattknoten
    'max_features': ['auto', 'sqrt'], # Anzahl der Merkmale, die bei jeder Aufteilung betrachtet werden
    'bootstrap': [True, False]        # Ob Bootstrap-Proben verwendet werden, wenn Bäume gebaut werden
}

# Grid Search mit Cross-Validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train.ravel())

# Beste Parameter und bestes Ergebnis anzeigen
print("Best Parameter: ", grid_search.best_params_)
print("Best Punktzahl (neg. MSE): ", grid_search.best_score_)

# Beste Modellvorhersagen treffen
best_model = grid_search.best_estimator_
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# Ergebnisse anzeigen
print("X_train:", X_train.ravel())
print("y_train:", y_train.ravel())
print("Train Predictions:", train_predictions)
print("X_test:", X_test.ravel())
print("y_test:", y_test.ravel())
print("Test Predictions:", test_predictions)

# MSE auf Trainings- und Testdaten
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
print("Mean Squared Error best model on traindata:", train_mse)
print("Mean Squared Error best model on testdata:", test_mse)

# Visualisierung der tatsächlichen Daten und Vorhersagen für Trainingsdaten
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, label='Train Actual Data', marker='o')
plt.plot(range(len(train_predictions)), train_predictions, label='Train Predictions', marker='x')
plt.title('Train Data: Actual vs Predictions')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualisierung der tatsächlichen Daten und Vorhersagen für Testdaten
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Test Actual Data', marker='o')
plt.plot(range(len(test_predictions)), test_predictions, label='Test Predictions', marker='x')
plt.title('Test Data: Actual vs Predictions')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Verbesserung Grid Search visualisieren
results = pd.DataFrame(grid_search.cv_results_)
mean_test_scores = results['mean_test_score'] * -1  # neg_mean_squared_error umkehren

plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_estimators'], mean_test_scores[:len(param_grid['n_estimators'])], marker='o')
plt.title('Grid Search Progress: Mean Test Score vs n_estimators')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Mean Test Score (MSE)')
plt.grid(True)
plt.show()
