import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class AttackModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = RandomForestClassifier()
        self.accuracy = None
        self.cm = None
        self._train()

    def _train(self):
        df = pd.read_csv(self.file_path)

        # ⚠️ Adjust this based on your dataset
        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.cm = confusion_matrix(y_test, y_pred)

        self.sample_input = X_test.iloc[0]

    def predict(self):
        pred = self.model.predict([self.sample_input])[0]
        return pred