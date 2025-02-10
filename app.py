from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from views import views
import subprocess
import os
import pathlib
import textwrap
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")

# Configure paths securely
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'E:/aircanvas/images/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini safely
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') or "AIzaSyAxPpQ5V-tiOx6TwH7yGpfwYhut5-Kf0PI"
genai.configure(api_key=GOOGLE_API_KEY)

def to_markdown(text):
    text = text.replace('*', ' *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

@app.route('/')
def index():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'paint_drawing.png')
    if pathlib.Path(image_path).exists():
        return render_template('index.html', image_filename='paint_drawing.png')
    else:
        return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        user_prompt = request.form['user_prompt']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'paint_drawing.png')
        
        if not pathlib.Path(image_path).exists():
            return abort(404, description="Image not found")
        
        with Image.open(image_path) as img:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content([user_prompt, img])  # Removed stream=True
            
            if not response.parts:
                return jsonify({"error": "No response generated"}), 500
                
            response_text = to_markdown(response.text)
            return render_template('index.html', 
                                image_filename='paint_drawing.png',
                                response_text=response_text)
    
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        return abort(500, description=str(e))

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/run_Aircanvas')
def run_Aircanvas():
    try:
        subprocess.Popen(['python', 'Aircanvas.py'])
        return jsonify({"status": "Aircanvas started"})
    except Exception as e:
        print(f"Error starting Aircanvas: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)