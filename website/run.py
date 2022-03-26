from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

def model():
    return render_template('analyze.html')

if __name__ == '__main__':
     app.run(debug=True)
