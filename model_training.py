import pandas as pd

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")

X = df.drop("label",axis=1)
y = df["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

models = {

"RandomForest":RandomForestClassifier(),

"DecisionTree":DecisionTreeClassifier(),

"LogisticRegression":LogisticRegression(max_iter=200)

}

for name,model in models.items():

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    acc=accuracy_score(y_test,preds)

    print(name,"Accuracy:",acc)