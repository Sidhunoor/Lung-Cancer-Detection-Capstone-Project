import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset into a pandas dataframe
df = pd.read_csv('C:/Users/shubh/Downloads/survey lung cancer.csv')


# Check the dataset
print(df.head())

# Check the shape of the dataset
print(df.shape)

# Check the data types of the variables
print(df.dtypes)

# Check the number of missing values
print(df.isna().sum())

# Check the distribution of lung cancer cases
sns.countplot(x='LUNG_CANCER', data=df)
plt.title('Distribution of Lung Cancer Cases')
plt.show()

# Check the correlation between age and lung cancer
sns.boxplot(x='LUNG_CANCER', y='AGE', data=df)
plt.title('Age vs. Lung Cancer')
plt.show()

# Check the association between smoking and lung cancer
smoking_counts = df.groupby(['LUNG_CANCER', 'SMOKING']).size().unstack()
smoking_counts.plot(kind='bar', stacked=True)
plt.title('Lung Cancer by Smoking Status')
plt.show()

# Check the differences in lung cancer between genders
gender_counts = df.groupby(['LUNG_CANCER', 'GENDER']).size().unstack()
gender_counts.plot(kind='bar', stacked=True)
plt.title('Lung Cancer by Gender')
plt.show()


# Histogram of Age
plt.hist(df['AGE'], bins=20)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar chart of smoking status
smoking_counts = df['SMOKING'].value_counts()
plt.bar(smoking_counts.index, smoking_counts.values)
plt.title('Smoking Status')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

# Scatter plot of age and anxiety levels
plt.scatter(df['AGE'], df['ANXIETY'])
plt.title('Age vs Anxiety')
plt.xlabel('Age')
plt.ylabel('Anxiety')
plt.show()

# Box plot of chronic disease status
sns.boxplot(x='CHRONIC DISEASE', y='AGE', data=df)
plt.title('Chronic Disease Status')
plt.xlabel('Chronic Disease')
plt.ylabel('Age')
plt.show()

# Pairplot of symptoms and lung cancer status
sns.pairplot(df, vars=['COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'], hue='LUNG_CANCER')
plt.title('Symptoms and Lung Cancer')
plt.show()



# Drop the "GENDER" variable since it is not useful for classification
df.drop('GENDER', axis=1, inplace=True)

# Convert categorical variables to numeric using label encoding
le = LabelEncoder()
df['SMOKING'] = le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS'] = le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY'] = le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE'] = le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE'] = le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE'] = le.fit_transform(df['FATIGUE'])
df['ALLERGY'] = le.fit_transform(df['ALLERGY'])
df['WHEEZING'] = le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING'] = le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING'] = le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH'] = le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY'] = le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN'] = le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

# Split the dataset into training and testing sets
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))


