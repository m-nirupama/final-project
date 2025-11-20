from flask import Flask, render_template, request, redirect, url_for, session, flash
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import io
import base64
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from googletrans import Translator
from utils.model import ResNet9
from utils.disease import disease_dic

# ============================================
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ============================================
# Disease labels and model setup
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# ============================================
# Prediction Function
def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.5)
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()], image

# ============================================
# Database Initialization
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, email TEXT, password TEXT)')
    conn.close()

init_db()

# ============================================
# Routes

# Landing Page
@app.route('/')
def landing():
    return render_template('landing.html', title='Welcome to Smart Farm')

# Home Page (after login)
@app.route('/home')
def home():
    if 'user' not in session:
        flash("Please login to access the app", "warning")
        return redirect(url_for('login'))
    return render_template('index.html', title='Smart Farm - Home')

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']

        if password != confirm:
            flash("Passwords do not match", "danger")
            return redirect(url_for('signup'))

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            flash("Email already exists", "warning")
            return redirect(url_for('signup'))

        hashed = generate_password_hash(password)
        cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
        conn.commit()
        conn.close()

        flash("Signup successful. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html', title='Signup')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT username, password FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user'] = user[0]
            flash(f"Welcome {user[0]}", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password", "danger")

    return render_template('login.html', title='Login')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('landing'))

# Disease Prediction
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if 'user' not in session:
        flash("Login required to continue", "warning")
        return redirect(url_for('login'))

    translator = Translator()
    title = 'MyCrop - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)

        try:
            img_bytes = file.read()
            prediction, pil_img = predict_image(img_bytes)

            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG')
            image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

            prediction_text = str(disease_dic.get(prediction, "Information not found."))

            cause_marker = "Cause of disease:"
            prevention_marker = "How to prevent/cure the disease"
            cause_start = prediction_text.find(cause_marker)
            prevention_start = prediction_text.find(prevention_marker)

            if cause_start == -1 or prevention_start == -1:
                cause_of_disease = "No specific cause information available."
                prevention_methods = "No specific prevention information available."
            else:
                cause_of_disease = prediction_text[cause_start + len(cause_marker):prevention_start].strip()
                prevention_methods = prediction_text[prevention_start + len(prevention_marker):].strip()

            target_lang = request.form.get('language') or 'en'
            if target_lang != 'en':
                try:
                    prediction = translator.translate(prediction, dest=target_lang).text
                    cause_of_disease = translator.translate(cause_of_disease, dest=target_lang).text
                    prevention_methods = translator.translate(prevention_methods, dest=target_lang).text
                except Exception as e:
                    print(f"Translation error: {e}")

            return render_template('disease-result.html',
                                   prediction=prediction,
                                   cause_of_disease=cause_of_disease,
                                   prevention_methods=prevention_methods,
                                   image_data=image_data,
                                   title=title)

        except Exception as e:
            print("General Error:", e)
            return render_template('disease.html', title=title)

    return render_template('disease.html', title=title)

# ============================================
if __name__ == '__main__':
    app.run(debug=True, port=5400)
