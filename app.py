from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask import jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__,static_folder='static')

# Secret key for sessions
app.secret_key = 'fe7022fd790d9bbf803ecd723c8e77f2'



# MongoDB setup
app.config["MONGO_URI"] = "mongodb://localhost:27017/usersdb"
mongo = PyMongo(app)




# Load the model 
model = joblib.load(r'C:\Users\abish\OneDrive\Desktop\clinics\Project\MediHelp\cardio_model.pkl')
dia_model = joblib.load(r'C:\Users\abish\OneDrive\Desktop\clinics\Project\MediHelp\diabetes_model.pkl')
resp_model = joblib.load(r'C:\Users\abish\OneDrive\Desktop\clinics\Project\MediHelp\respiratory_predict.pkl')  



# Home route
@app.route('/')
def index():
    return render_template('index.html')

    
# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        # flash('Please log in first!', 'error')
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', username=session['user'])


@app.route('/cardio')
def cardio():
    return render_template('cardio.html')


@app.route('/respiratory')
def respiratory():
    return render_template('resp.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes_predict.html')


@app.route('/info')
def info():
    if 'user' not in session:
        # flash('Please log in first!', 'error')
        return redirect(url_for('login'))
    return render_template('info.html',username=session['user'])


# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    # flash('You have been logged out.', 'info')
    return redirect(url_for('index'))




#USER AUTHENTICATION

# Sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        flash('You are already logged in!', 'info')
        return redirect(url_for('index'))  # Redirect to the dashboard if logged in
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the user or email already exists
        user = mongo.db.user.find_one({'$or': [{'username': username}, {'email': email}]})
        if user:
            flash('Username or Email already exists!', 'error')
            return redirect(url_for('signup'))
        
        # Hash the password before storing it
        hashed_password = generate_password_hash(password)
        
        # Insert the new user into the "users" collection
        mongo.db.user.insert_one({'username': username, 'email': email, 'password': hashed_password})
        
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')



#login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        flash('You are already logged in!', 'info')
        return redirect(url_for('index'))  
    
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        password = request.form['password']
        
        # Check if the user exists using either username or email
        user = mongo.db.user.find_one({'$or': [{'username': username_or_email}, {'email': username_or_email}]})
        
        if user and check_password_hash(user['password'], password):
            # Set session for the logged-in user
            session['user'] = user['username'] 
            flash('Login successful!', 'success')
            return redirect(url_for('index')) 
        else:
            flash('Invalid username/email or password!', 'error')
            return redirect(url_for('login')) 
    
    return render_template('login.html')




#MODEL PREDICTION


#diabetes_prediction
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get the data from the POST request (JSON format)
        data = request.get_json()

        # Extract input values from the received JSON
        gender = int(data['gender'])
        age = float(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        smoking_history = int(data['smoking_history'])
        bmi = float(data['bmi'])
        hba1c_level = float(data['HbA1c_level'])
        blood_glucose_level = float(data['blood_glucose_level'])

        # Calculate glucose_bmi_ratio
        glucose_bmi_ratio = blood_glucose_level / bmi

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, 
                                    bmi, hba1c_level, blood_glucose_level, glucose_bmi_ratio]],
                                  columns=['gender', 'age', 'hypertension', 'heart_disease', 
                                           'smoking_history', 'bmi', 'HbA1c_level', 
                                           'blood_glucose_level', 'glucose_bmi_ratio'])

        # Make the prediction
        prediction = dia_model.predict(input_data)

        # Return prediction result
        if prediction[0] == 1:
            return jsonify({'result': 'High diabetes risk'})
        else:
            return jsonify({'result': 'Low diabetes risk'})

    except Exception as e:
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500




#cardio vascular disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    # input data from the request (we expect a JSON body)
    data = request.get_json()
    print(f"Received data: {data}")

    try:
        # Extract individual values from the incoming data and convert to the correct type
        user_id = int('8')  
        age = int(data['age']) 
        gender = int(data['gender'])  
        height = int(data['height']) 
        weight = float(data['weight'])  
        ap_hi = int(data['ap_hi'])  
        ap_lo = int(data['ap_lo'])  
        cholesterol = int(data['cholesterol']) 
        gluc = int(data['gluc']) 
        active = int(data['active'])  

       
        print(f"Prepared input data for prediction: {[user_id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, active]}")

        #input data for prediction (in NumPy array format)
        input_data = np.array([[user_id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, active]])

        # Make the prediction using the model
        prediction = model.predict(input_data)
        print(f"Prediction: {prediction}")
        # Extract prediction value (assuming it's a list or array)
        prediction_value = prediction[0]

        # Send the result back as JSON
        if prediction_value == 1:
            return jsonify({'result': 'High cardio risk'})
        else:
            return jsonify({'result': 'Low cardio risk'})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500
    



#respiratory disease prediction
@app.route('/predict_respiratory', methods=['POST'])
def predict_respiratory():
    # Get the input data from the request (we expect a JSON body)
    data = request.get_json()
    print(f"Received data: {data}")

    try:
        # Extract individual values from the incoming data and convert to the correct type
        age = int(data['age'])  
        smoking = int(data['smoking'])  
        yellow_fingers = int(data['yellow_fingers']) 
        anxiety = int(data['anxiety'])  
        peer_pressure = int(data['peer_pressure'])
        chronic_disease = int(data['chronic_disease'])
        fatigue = int(data['fatigue']) 
        allergy = int(data['allergy']) 
        wheezing = int(data['wheezing'])  
        alcohol_consuming = int(data['alcohol_consuming'])  
        coughing = int(data['coughing']) 
        swallowing_difficulty = int(data['swallowing_difficulty']) 
        chest_pain = int(data['chest_pain']) 

        # prepared data for prediction
        print(f"Prepared input data for prediction: {[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain]}")

        # input data for prediction (in NumPy array format)
        input_data = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain]])

        # prediction using the model
        prediction = resp_model.predict(input_data)
        print(f"Prediction: {prediction}")
        # Extract prediction value (assuming it's a list or array)
        prediction_value = prediction[0]

        # Send the result back as JSON
        if prediction_value == 0:
            return jsonify({'prediction': 'High respiratory risk'})
        else:
            return jsonify({'prediction': 'Low respiratory risk'})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

