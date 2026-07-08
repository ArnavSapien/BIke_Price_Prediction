from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from joblib import load

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "updated_model.lb"

application = Flask(__name__)
app = application

model = load(MODEL_PATH)

BRAND_MAP = {
    "Royal Enfield": 1,
    "KTM": 2,
    "Bajaj": 3,
    "Harley": 4,
    "Yamaha": 5,
    "Honda": 6,
    "Suzuki": 7,
    "TVS": 8,
    "Kawasaki": 9,
    "Hyosung": 10,
    "Benelli": 11,
    "Mahindra": 12,
    "Triumph": 13,
    "Ducati": 14,
    "BMW": 15,
}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        kms_driven = int(request.form["Kms_Driven"])
        owner = int(request.form["owner"])
        age = int(request.form["age"])
        power = int(request.form["power"])
        brand_name = request.form["brand_name"]

        brand = BRAND_MAP.get(brand_name)
        if brand is None:
            return render_template("index.html", error_text="Please select a valid bike brand.")

        input_data = pd.DataFrame(
            [[kms_driven, owner, age, power, brand]],
            columns=["kms_driven", "owner", "age", "power", "brand"],
        )

        prediction = model.predict(input_data)[0]
        prediction_text = f"₹ {prediction:,.0f}"

        return render_template("index.html", prediction_text=prediction_text)

    except ValueError:
        return render_template("index.html", error_text="Please enter valid numeric values.")
    except Exception as e:
        return render_template("index.html", error_text=f"Something went wrong: {e}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
