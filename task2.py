import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('fraudTest.csv')


print(data.head())


print(data.isnull().sum())

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

X = X.drop(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num'], axis=1)


categorical_cols = ['merchant', 'category', 'gender']
numerical_cols = ['amt', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']


numerical_transformer = StandardScaler()


categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


log_reg = LogisticRegression()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()


models = {
    'Logistic Regression': log_reg,
    'Decision Tree': tree,
    'Random Forest': forest
}

for model_name, model in models.items():
   
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    clf.fit(X_train, y_train)
    
    
    y_pred = clf.predict(X_test)
    
  
    print(f"\n{model_name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
  
    if model_name == 'Random Forest':
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
