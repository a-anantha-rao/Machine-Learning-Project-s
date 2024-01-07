from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


app = Flask(__name__)


# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'
X = data[features].values
y = data[target].values

# Map the labels to numerical values
target_mapping = {
    'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
    'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
    'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
    'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
    'jute': 20, 'coffee': 21
}
reverse_mapping = {v: k for k, v in target_mapping.items()}

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y)

svm = SVC()
svm.fit(X, y)

# Initialize the SVM classifier with numerical labels
svm_numerical = SVC()
svm_numerical.fit(X, y)



admin_email = "admin123@gmail.com"
admin_password = "12345"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('lavanya.html')
@app.route('/about')
def about():
    return render_template('aboutus.html')
@app.route('/second')
def second():
    return render_template('prediction.html')

@app.route('/third')
def third():
    return render_template('acc.html')
@app.route('/forgot')
def forgot():
    return render_template('forgot.html')
@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/fourth')
def fourth():
    return render_template('abouts.html')
@app.route('/comp')
def comp():
    return render_template('comp.html')
@app.route('/fifth')
def fifth():
    return render_template('lavmain.html')

@app.route('/sixth')
def sixth():
    return render_template('metrices.html')


@app.route('/signin', methods=['POST'])
def signin():
    email = request.form.get('email')
    password = request.form.get('password')

    if email == admin_email:
        if password == admin_password:
            return render_template('lavmain.html')
        else:
            return "Wrong password. Please try again."
    else:
        return "Email not registered. Please contact the administrator."


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/result', methods=['POST'])
def make_prediction():
    # Get user input from the HTML form
    n = int(request.form['nValue'])
    p = int(request.form['pValue'])
    k = int(request.form['kValue'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['rainfall'])
    algorithm = request.form['algorithm']
    
    # Create a list from the user input
    crop_data = [n, p, k, temperature, humidity, ph, rainfall]
    
    if algorithm == 'DecisionTree':
        user_input = np.array([crop_data])
        # Predict using the Decision Tree model
        predicted_label = decision_tree.predict(user_input)
        predicted_crop = reverse_mapping[predicted_label[0]]
        return render_template('resultdecision.html', prediction_result=predicted_crop)
 
    if algorithm == 'SVM':
        user_input = np.array([crop_data])
        # Predict using the SVM model
        predicted_label = svm_numerical.predict(user_input)
        predicted_crop = reverse_mapping[predicted_label[0]]
        return render_template('resultsvm.html', prediction_result=predicted_crop)


if __name__ == '__main__':
    app.run(debug=True)
