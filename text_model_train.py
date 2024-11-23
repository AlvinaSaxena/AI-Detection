import pandas as pd
import numpy as np
from transformers import BertTokenizer 
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split

def tokenize_text(text, tokenizer):
    return tokenizer(
        text,
        padding='max_length', 
        max_length=128,        
        truncation=True,       
        return_tensors="tf"    
    )

def predict_text(text, tokenizer, model):
    encoding = tokenize_text(text,tokenizer)
    input_id = tf.convert_to_tensor([encoding['input_ids'][0]])
    attention_mask = tf.convert_to_tensor([encoding['attention_mask'][0]])
    logits = model.predict([input_id, attention_mask]).logits
    confidence = tf.nn.softmax(logits)[0].numpy()
    confidence = max([float(c)*100 for c in confidence])
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    return "AI-generated" if prediction == 1 else "Human-written", confidence

if __name__ == "__main__":

    df = pd.read_csv('data/Training_Essay_Data.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []
    for text in df['text']:
        encoding = tokenize_text(text)
        input_ids.append(encoding['input_ids'][0])
        attention_masks.append(encoding['attention_mask'][0])

    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)
    labels = tf.convert_to_tensor(df['generated'].values)

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_labels, val_labels = train_test_split(
        np.array(input_ids), np.array(attention_masks), np.array(labels), test_size=0.2, random_state=42
    )

    history = model.fit(
        [train_input_ids, train_attention_masks],
        train_labels,
        validation_data=([val_input_ids, val_attention_masks], val_labels),
        epochs=1,      
        batch_size=16  
    )

    loss, accuracy = model.evaluate([val_input_ids, val_attention_masks], val_labels)
    print(f"Validation Accuracy: {accuracy:.2f}")


    print(predict_text("Your new sample text here"))

    model.save_pretrained("fine_tuned_bert_model")
    tokenizer.save_pretrained("fine_tuned_bert_model")