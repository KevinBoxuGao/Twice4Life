from flask import Flask
from models import generate_song, lstm_model
import torch

app = Flask('twice')
model = lstm_model(1024, 4)

@app.route('/')
def generate():
    model.load_state_dict(torch.load('models/generator_0.pth'))
    generate_song(model, "generated/t4l.wav")
    return send_from_directory('generated', 't4l.wav')

