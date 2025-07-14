from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load models
MODEL_NAME_1 = "textattack/bert-base-uncased-SST-2"
MODEL_NAME_2 = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_NAME_1)
model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_1)
model_1.eval()

tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_NAME_2)
model_2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_2)
model_2.eval()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "Text is required"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Input text is empty"}), 400

        # Get sentiment analysis from both models
        result_1 = analyze_sentiment(text, tokenizer_1, model_1, ["negative", "positive"])
        result_2 = analyze_sentiment(text, tokenizer_2, model_2, ["negative", "neutral", "positive"])

        if result_1 is None or result_2 is None:
            return jsonify({"error": "Sentiment analysis failed"}), 500

        # Extract probabilities
        neg1 = result_1['probabilities'].get('negative', 0)
        pos1 = result_1['probabilities'].get('positive', 0)
        neg2 = result_2['probabilities'].get('negative', 0)
        neu2 = result_2['probabilities'].get('neutral', 0)  # Neutral only from model 2
        pos2 = result_2['probabilities'].get('positive', 0)

        # Sum probabilities (before normalization)
        total_negative = neg1 + neg2
        total_neutral = neu2  # Only model 2 provides neutral
        total_positive = pos1 + pos2

        # Normalize to make sure sum is 100%
        total_sum = total_negative + total_neutral + total_positive
        if total_sum > 0:  # Avoid division by zero
            avg_negative = round((total_negative / total_sum) * 100, 2)
            avg_neutral = round((total_neutral / total_sum) * 100, 2)
            avg_positive = round((total_positive / total_sum) * 100, 2)
        else:
            avg_negative, avg_neutral, avg_positive = 0, 0, 0

        # Ensure total is exactly 100% by adjusting the largest value
        adjustment = 100 - (avg_negative + avg_neutral + avg_positive)
        if adjustment != 0:
            if avg_positive >= avg_negative and avg_positive >= avg_neutral:
                avg_positive += adjustment
            elif avg_negative >= avg_positive and avg_negative >= avg_neutral:
                avg_negative += adjustment
            else:
                avg_neutral += adjustment

        # Final sentiment decision
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            final_sentiment = "positive"
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            final_sentiment = "negative"
        else:
            final_sentiment = "neutral"

        return jsonify({
            'sentiment': final_sentiment,
            'probabilities': {
                'negative': avg_negative,
                'neutral': avg_neutral,
                'positive': avg_positive
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_sentiment(text, tokenizer, model, labels):
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)

        scores = output.logits[0].tolist()
        probabilities = softmax(scores)  # Ensures valid probability values

        # Convert to dictionary format
        sentiment_result = {
            'probabilities': {labels[i]: round(probabilities[i] * 100, 2) for i in range(len(labels))}
        }

        return sentiment_result
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return None  # Ensure no crashes

if __name__ == '__main__':
    app.run(debug=True)  # Running on http://localhost:5000
