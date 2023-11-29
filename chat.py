from tkinter import *
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

class ChatBotApplication:
    def __init__(self):
        self.window = Tk()
        self.setup_chatbot_window()

    def run(self):
        self.window.mainloop()

    def setup_chatbot_window(self):
        self.window.title("ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=700, height=600, bg='#ffffff')

        head_label = Label(self.window, bg='#295b5e', fg='#ffffff', text="Welcome", font='Courier 13 bold', pady=10)
        head_label.place(relwidth=1)

        line = Label(self.window, width=450, bg='#295b5e')
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        self.text_widget = Text(self.window, width=20, height=2, bg='#295b5e', fg='#ffffff', font='Courier 14', padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scroll_bar = Scrollbar(self.text_widget)
        scroll_bar.place(relheight=1, relx=0.974)
        scroll_bar.configure(command=self.text_widget.yview)

        bottom_label = Label(self.window, bg='#D9D9D9', height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        self.msg_entry = Entry(bottom_label, bg='#295b5e', fg='#ffffff', font='Courier 14')
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)

        send_button = Button(bottom_label, text="Send", font='Courier 13 bold', width=20, bg='#295b5e', command=lambda: self.on_enter(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def on_enter(self, event):
        msg = self.msg_entry.get()
        self.insert_message(msg, "You")

    def insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        response = self.get_chatbot_response(msg)
        msg2 = f"{bot_name}: {response}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def get_chatbot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

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

if __name__ == "__main__":
    app = ChatBotApplication()
    app.run()
