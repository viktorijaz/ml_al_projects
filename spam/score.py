"""
Rung 1 — spam/ · score a message with the trained model.

Usage:
    python score.py "Congrats! You won a FREE prize, text WIN to 80085"
    echo "hey are we still on for lunch" | python score.py
"""

import sys
from pathlib import Path

import joblib

MODEL = Path(__file__).parent / "model.joblib"

if not MODEL.exists():
    sys.exit("No model found. Train it first:  python train.py")

# Message from CLI args, or piped via stdin.
msg = " ".join(sys.argv[1:]).strip() or sys.stdin.read().strip()
if not msg:
    sys.exit('Give me a message:  python score.py "your text here"')

model = joblib.load(MODEL)

# P(spam); 0.5 is the default threshold (revisited in rung 2 — report-card/).
proba_spam = model.predict_proba([msg])[0][1]
label = "SPAM" if proba_spam >= 0.5 else "HAM"

print(f"\n  {label}   (P(spam) = {proba_spam:.1%})")
print(f"  message: {msg[:80]}{'...' if len(msg) > 80 else ''}\n")
