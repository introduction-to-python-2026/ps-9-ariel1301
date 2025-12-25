# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()
df = df.dropna()
import seaborn as sns
sns.pairplot(df, hue='NHR')
selected_features = ['NHR', 'RPDE']
target_feature = 'status'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_for_analysis = scaler.fit_transform(df[selected_features])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_for_analysis, df[target_feature], test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(model, 'ari01.joblib')
