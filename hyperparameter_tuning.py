from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200],  # instead of 100,200,300,400...
    'max_depth': [10, 20]        # instead of 10,20,30...
}

model = RandomForestClassifier()
grid = GridSearchCV(model, param_grid, cv=2, n_jobs=-1)  