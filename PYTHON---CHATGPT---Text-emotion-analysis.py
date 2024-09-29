from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load emotion detection model
emotion_model = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

@app.route('/analyze_emotions', methods=['POST'])
def analyze_emotions():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "Please provide text for emotion analysis."}), 400
    
    text = data['text']
    
    # Analyze the emotions
    emotions = emotion_model(text)
    
    return jsonify(emotions)

if __name__ == '__main__':
    app.run(debug=True)
