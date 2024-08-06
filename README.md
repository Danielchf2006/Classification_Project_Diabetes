# Diabetes Classification w/ KNN

The Purpose: Creating a K-Nearest Neighbor Algorithm to identify cases of potential diabetes carriers. 

## Test Environment

Python 3.9.6

## Usage
Run the script using:
```python
python3 main.py
```
### Installation
To run this project, you need to have Python installed along with the following libraries:
numpy, pandas, scikit-learn

You can import these libraries through Pip.

### Load Dataframe

The National Institute of Diabetes and Digestive and Kidney Diseases provides the dataset.
The dataset used is the Pima Indians Diabetes Database, which contains 768 entries with 8 features and a target variable indicating whether the patient has diabetes (1) or not (0).
The input features include:
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age



The purpose of 'pd.read_csv('diabetes.csv')' is to Specify the path to the CSV file to be loaded. Ensure the file path is correct and accessible from the script's location.
```python
Diabetes_dataset = pd.read_csv('diabetes.csv')
```

### Prepare Data for Modeling

There are cases within the dataset that have 0 as inputs for the features 'Glucose,' 'Blood Pressure,' 'SkinThickness,' 'Insulin,' and 'BMI.' Therefore, they need to be replaced as having 0 in any of those categories would be deemed as dead. 
```Python
for var in vars:
    Diabetes_dataset[var]= Diabetes_dataset[var].replace(0, np.nan) #no data there
    mean = int(Diabetes_dataset[var].mean(skipna=True)) #Pandas command for all columns that's been skipped/ replace with the average instead of deleting.
    Diabetes_dataset[var] = Diabetes_dataset[var].replace(np.nan, mean)
```
By replacing 0 with NaN, we can then replace those NaN with the mean of other cases so that that case is still applicable during the data training process since other variables are still valuable. 


### Training the data 
We can start by splitting the data into training sets and testing sets. 
```Python 
X = Diabetes_dataset.iloc[:, 0:8] #Panda for splitting
Y= Diabetes_dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
#### Eliminating Bias
Using Standardscaler() from the Sklearn library, we can eliminate potential bias and scale variables into appropriate sizes
```Python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #make sure there is no bias and scale all of the data. Only training the X because it is the input
```

Now, we can implement the KNN algorithm in training. The model used is the K-Nearest Neighbors (KNN) classifier with the following parameters:
n_neighbors=10: Number of neighbors to use.
p=2: The power parameter(P=2) corresponding to the Euclidean distance.
metric='euclidean': The distance metric to use.

```Python
Diabetes_classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='euclidean')
Diabetes_classifier.fit(X_train, y_train)
```

### Evaluation
The model's performance is evaluated using:

Confusion Matrix: A table used to describe the performance of a classification model.
Accuracy Score: The ratio of correctly predicted instances to the total instances.

```Python
y_pred= Diabetes_classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred) #predicted across the top and actual going down
```

