import numpy as np
import joblib

class LSTMWrapper:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = np.zeros((1, self.max_length), dtype=int)
        padded[:, -len(sequences[0]):] = sequences[0][-self.max_length:]
        return padded

    def predict(self, text):
        input_data = self.preprocess(text)
        prob = self.model.predict(input_data, verbose=0)[0][0]
        label = "Fake" if prob < 0.5 else "Real"
        confidence = prob if label == "Real" else 1 - prob
        return label, confidence
