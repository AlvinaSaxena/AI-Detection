from flask import Flask, render_template, request, redirect,url_for
from text_model_train import predict_text
from transformers import TFBertForSequenceClassification, BertTokenizer

model = TFBertForSequenceClassification.from_pretrained("./text_model")
tokenizer = BertTokenizer.from_pretrained("./text_model")

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

@app.route('/predictText/', methods=['POST'])
def predictText():
    text = request.form.get('user_text')
    prediction, confidence = predict_text(text, tokenizer, model)
    return f"This text is {prediction} with the confidence {confidence:.2f}"

if __name__ == '__main__':
    app.run(debug=True, port=8000)
