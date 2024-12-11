import os
from flask import Flask, render_template, request, redirect,url_for
from text_model_train import predict_text
from image_model_train import predict_single_image, preprocess_image
from transformers import TFBertForSequenceClassification, BertTokenizer, ViTForImageClassification, ViTImageProcessor

text_model = TFBertForSequenceClassification.from_pretrained("./text_model")
image_model = ViTForImageClassification.from_pretrained("./image_model")

tokenizer = BertTokenizer.from_pretrained("./text_model")
processor = ViTImageProcessor.from_pretrained("./image_model")

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Path to the uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    prediction, confidence = predict_text(text, tokenizer, text_model)
    confidence = confidence*100
    result= f"This text is {prediction}"
    return render_template('checkText.html', result=result, confidence=confidence, prediction=prediction, text=text)

@app.route('/predictImage/', methods=['POST'])
def predictImage():
    if 'image' not in request.files:
        return 'No image file uploaded'
    image_file = request.files['image']

    if image_file.filename != '':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

    try:
        inputs = preprocess_image(image_file, processor)
        predicted_label, confidence = predict_single_image(
            image_file,
            image_model,
            processor,
            inputs
        )
        confidence = confidence*100
        result=f"The given image is {predicted_label}"
    except Exception as e:
        return str(e)
    image_url = url_for('static', filename=f'uploads/{image_file.filename}')
    return render_template('checkImage.html', result=result, confidence=confidence, predicted_label=predicted_label, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
