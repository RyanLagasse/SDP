from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle text generation
@app.route('/generate', methods=['POST'])
def generate_text():
    user_input = request.form['user_input']  # Get user input from the form
    inputs = tokenizer.encode(user_input, return_tensors='pt')  # Tokenize input
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)  # Generate text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output
    return render_template('index.html', user_input=user_input, response=response)

if __name__ == '__main__':
    app.run(debug=True)
