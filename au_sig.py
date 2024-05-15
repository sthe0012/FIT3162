import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('D:/Monash/semester 6/FIT 3162/Mexp_Features/au_c.csv')

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

feature_importance_df = pd.DataFrame({'AU': X.columns, 'Importance': clf.feature_importances_})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
feature_importance_df.to_csv('D:/Monash/semester 6/FIT 3162/au_c_importance.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['AU'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('AU')
plt.title('Action unit importance')
plt.gca().invert_yaxis()
plt.show()
