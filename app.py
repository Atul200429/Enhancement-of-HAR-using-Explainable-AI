from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load model & scaler (update paths if needed) ──────────────────────────────
MODEL_PATH  = 'Human_activity_recognition_model2.pkl'
SCALER_PATH = 'scaler.pkl'          # you'll save this from Colab (see README)
TEST_CSV    = 'test.zip.csv'        # kept for "Try Random Sample" feature

model  = joblib.load(MODEL_PATH)  if os.path.exists(MODEL_PATH)  else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
test_df = pd.read_csv(TEST_CSV).drop('Activity', axis=1) if os.path.exists(TEST_CSV) else None

CLASSES = ['LAYING', 'SITTING', 'STANDING', 'WALKING',
           'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

ICONS = {
    'LAYING'             : '🛌',
    'SITTING'            : '🪑',
    'STANDING'           : '🧍',
    'WALKING'            : '🚶',
    'WALKING_DOWNSTAIRS' : '⬇️',
    'WALKING_UPSTAIRS'   : '⬆️',
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found. Please check app.py path.'}), 500

    try:
        # ── CSV file upload ───────────────────────────────────────────────────
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            df   = pd.read_csv(file)
            # drop Activity column if present
            if 'Activity' in df.columns:
                df = df.drop('Activity', axis=1)
            features = df.iloc[0].values.reshape(1, -1)

        # ── Random sample from test set ───────────────────────────────────────
        elif request.form.get('random') == 'true':
            if test_df is None:
                return jsonify({'error': 'test.zip.csv not found in app folder.'}), 500
            idx      = np.random.randint(0, len(test_df))
            features = test_df.iloc[idx].values.reshape(1, -1)

        else:
            return jsonify({'error': 'No input provided.'}), 400

        # ── Scale if scaler available ─────────────────────────────────────────
        if scaler is not None:
            features = scaler.transform(features)

        # ── Predict ───────────────────────────────────────────────────────────
        proba       = model.predict_proba(features)[0]
        pred_idx    = int(np.argmax(proba))
        pred_label  = CLASSES[pred_idx]
        confidence  = round(float(proba[pred_idx]) * 100, 2)

        scores = [
            {'label': CLASSES[i], 'icon': ICONS[CLASSES[i]],
             'prob': round(float(p) * 100, 2)}
            for i, p in enumerate(proba)
        ]
        scores.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            'prediction' : pred_label,
            'icon'       : ICONS[pred_label],
            'confidence' : confidence,
            'scores'     : scores,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
