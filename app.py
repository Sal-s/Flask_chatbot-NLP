from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
history = []  # Create an empty list to store chat history

# Load data and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sals"

@app.route('/')
def index():
    return render_template('index.html', history=history)  # Pass chat history to the template

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    history.append({'user': user_input, 'bot': get_chatbot_response(user_input)})  # Add user input and bot response to history
    response = history[-1]['bot']  # Get the latest bot response from history
    return jsonify({'bot_response': response, 'history': history})  # Send bot response and updated history to the client

def get_chatbot_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

if __name__ == '__main__':
    app.run(debug=True)
