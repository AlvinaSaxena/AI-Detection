from flask import Flask, render_template, request, redirect,url_for
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/contact/')
def contact():
    return render_template('contact.html')

@app.route('/textModel/')
def textModel():
    return render_template('textModel.html')

@app.route('/imageModel/')
def imageModel():
    return render_template('imageModel.html')

@app.route('/checkText/')
def checkText():
    return render_template('checkText.html')

@app.route('/checkImage/')
def checkImage():
    return render_template('checkImage.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
