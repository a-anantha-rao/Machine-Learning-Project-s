from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('air.csv')

# Extract the input features and target variable
X = data[['AQI Value', 'CO AQI Value', 'CO AQI Category', 'Ozone AQI Value', 'Ozone AQI Category', 'NO2 AQI Value', 'NO2 AQI Category', 'PM2.5 AQI Value']]
y = data['PM2.5 AQI Category']

# Label encode categorical variables
label_encoder = LabelEncoder()
X['CO AQI Category'] = label_encoder.fit_transform(X['CO AQI Category'])
X['Ozone AQI Category'] = label_encoder.fit_transform(X['Ozone AQI Category'])
X['NO2 AQI Category'] = label_encoder.fit_transform(X['NO2 AQI Category'])

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

# Create a K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X, y)

# Create an SVM Classifier
svm_classifier = SVC()
svm_classifier.fit(X, y)

# Define the mapping for predictions
label_mapping = {
    0: 'Good',
    1: 'Moderate',
    2: 'Unhealthy',
    3: 'Unhealthy for Sensitive Area',
    4: 'Hazardous'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/two')
def two():
    return render_template('signin.html')

@app.route('/met')
def met():
    return render_template('meti.html')

@app.route('/comp')
def comp():
    return render_template('comp.html')

@app.route('/four')
def four():
    return render_template('accuracy.html')

@app.route('/three')
def three():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city']
    aqi = float(request.form['aqi'])
    co_aqi = float(request.form['co_aqi'])
    co_cat = request.form['co_cat']
    ozone_aqi = float(request.form['ozone_aqi'])
    ozone_cat = request.form['ozone_cat']
    no2_aqi = float(request.form['no2_aqi'])
    no2_cat = request.form['no2_cat']
    pm25_aqi = float(request.form['pm25_aqi'])

    # Make a prediction using Random Forest
    rf_prediction = rf_classifier.predict([[aqi, co_aqi, co_cat, ozone_aqi, ozone_cat, no2_aqi, no2_cat, pm25_aqi]])
    rf_prediction_label = label_mapping[rf_prediction[0]]

    return render_template('result_rf.html', city=city, prediction=rf_prediction_label)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    city = request.form['city']
    aqi = float(request.form['aqi'])
    co_aqi = float(request.form['co_aqi'])
    co_cat = request.form['co_cat']
    ozone_aqi = float(request.form['ozone_aqi'])
    ozone_cat = request.form['ozone_cat']
    no2_aqi = float(request.form['no2_aqi'])
    no2_cat = request.form['no2_cat']
    pm25_aqi = float(request.form['pm25_aqi'])

    co_cat_encoded = label_encoder.transform([co_cat])[0]
    ozone_cat_encoded = label_encoder.transform([ozone_cat])[0]
    no2_cat_encoded = label_encoder.transform([no2_cat])[0]

    # Make a prediction using K-Nearest Neighbors
    nb_prediction = knn_classifier.predict([[aqi, co_aqi, co_cat_encoded, ozone_aqi, ozone_cat_encoded, no2_aqi, no2_cat_encoded, pm25_aqi]])
    nb_prediction_label = label_mapping[nb_prediction[0]]

    return render_template('result_knn.html', city=city, prediction=nb_prediction_label)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    city = request.form['city']
    aqi = float(request.form['aqi'])
    co_aqi = float(request.form['co_aqi'])
    co_cat = request.form['co_cat']
    ozone_aqi = float(request.form['ozone_aqi'])
    ozone_cat = request.form['ozone_cat']
    no2_aqi = float(request.form['no2_aqi'])
    no2_cat = request.form['no2_cat']
    pm25_aqi = float(request.form['pm25_aqi'])

    co_cat_encoded = label_encoder.transform([co_cat])[0]
    ozone_cat_encoded = label_encoder.transform([ozone_cat])[0]
    no2_cat_encoded = label_encoder.transform([no2_cat])[0]

    # Make a prediction using SVM
    svm_prediction = svm_classifier.predict([[aqi, co_aqi, co_cat_encoded, ozone_aqi, ozone_cat_encoded, no2_aqi, no2_cat_encoded, pm25_aqi]])
    svm_prediction_label = label_mapping[svm_prediction[0]]

    return render_template('result_svm.html', city=city, prediction=svm_prediction_label)

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == 'admin123@gmail.com' and password == '1234':
            # Redirect to the dashboard or desired page after successful sign-in
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid email or password. Please try again.'
            return render_template('signin.html', error=error)

    return render_template('signin.html')

@app.route('/dashboard')
def dashboard():
    # Render the dashboard or desired page after successful sign-in
    return render_template('main.html')

if __name__ == '__main__':
    app.run()
