from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Przykładowe dane Iris:")
print(df.head())

print("\n📈 Informacje o danych:")
print(df.info())

print("\n📊 Statystyki opisowe:")
print(df.describe())

print("\n🧹 Brakujące wartości:")
print(df.isnull().sum())

df.hist(figsize=(8, 6))
plt.suptitle("Histogramy cech Iris")
plt.show()

sns.pairplot(df, hue='species', palette='husl')
plt.suptitle("Pairplot cech Iris", y=1.02)
plt.show()

# Definiowanie cech (X) i etykiet (y)
X = df.drop(columns=['species'])  # Wszystkie kolumny oprócz 'species'
y = df['species']  # Kolumna 'species' jako etykiety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Iris")
plt.show()
print(f"Final Accuracy: {accuracy:.2f}")
