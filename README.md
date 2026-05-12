# RxGuard

RxGuard is a prescription safety checker for reviewing drug combinations, patient risk factors, and known interaction patterns before medication decisions are made.

## Features

- Drug interaction checks for severe, moderate, and mild combinations
- Brand-to-generic drug name matching
- Pregnancy, allergy, age, and condition-based safety warnings
- Risk score for the selected medication list
- Prediction model trained on 99,000+ records
- Interaction graph and pairwise risk heatmap
- Exportable summary for review or printing

## Tech Stack

- Python
- Flask
- Pandas and NumPy
- Scikit-learn
- HTML, CSS, JavaScript, and D3.js

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000` in the browser.

## Deployment

Use `gunicorn app:app` as the start command. The app expects the CSV files in `data/` and the trained model at `models/best_model.pkl`.

## Note

This project is for educational and informational use. Medication decisions should always be reviewed by a licensed clinician or pharmacist.
