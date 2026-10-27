from flask import Flask, request, jsonify, render_template_string
import pickle

# Create Flask application
app = Flask(__name__)

# Load trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Names of Iris classes
class_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]


# =========================================================
# HTML WEB PAGE
# =========================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

    <title>Iris Flower ML Prediction</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {

            margin: 0;
            padding: 0;

            font-family: Arial, sans-serif;

            /* Gradient background */

            background: linear-gradient(
                135deg,
                #667eea,
                #764ba2,
                #f093fb
            );

            min-height: 100vh;

            display: flex;

            justify-content: center;

            align-items: center;
        }


        .container {

            width: 450px;

            background: rgba(
                255,
                255,
                255,
                0.95
            );

            padding: 35px;

            border-radius: 20px;

            box-shadow:
                0 15px 35px
                rgba(0, 0, 0, 0.3);

            text-align: center;
        }


        .icon {

            font-size: 55px;

            margin-bottom: 5px;
        }


        h1 {

            color: #4a148c;

            margin-bottom: 5px;
        }


        .subtitle {

            color: #666;

            margin-bottom: 25px;
        }


        .input-group {

            text-align: left;

            margin-bottom: 15px;
        }


        label {

            display: block;

            margin-bottom: 6px;

            color: #4a148c;

            font-weight: bold;
        }


        input {

            width: 100%;

            padding: 12px;

            border: 2px solid #d1c4e9;

            border-radius: 10px;

            font-size: 16px;

            outline: none;

            transition: 0.3s;
        }


        input:focus {

            border-color: #7b1fa2;

            box-shadow:
                0 0 8px
                rgba(123, 31, 162, 0.3);
        }


        button {

            width: 100%;

            padding: 14px;

            margin-top: 10px;

            border: none;

            border-radius: 10px;

            background: linear-gradient(
                90deg,
                #7b1fa2,
                #e91e63
            );

            color: white;

            font-size: 18px;

            font-weight: bold;

            cursor: pointer;

            transition: 0.3s;
        }


        button:hover {

            transform: scale(1.03);

            box-shadow:
                0 8px 20px
                rgba(123, 31, 162, 0.4);
        }


        .result {

            margin-top: 25px;

            padding: 15px;

            border-radius: 12px;

            background: linear-gradient(
                90deg,
                #43cea2,
                #185a9d
            );

            color: white;

            font-size: 21px;

            font-weight: bold;
        }


        .footer {

            margin-top: 20px;

            color: #777;

            font-size: 13px;
        }

    </style>

</head>


<body>


<div class="container">


    <div class="icon">
        🌸
    </div>


    <h1>
        Iris Flower Prediction
    </h1>


    <div class="subtitle">

        Machine Learning Model Prediction

    </div>


    <form
        action="/predict-web"
        method="post"
    >


        <div class="input-group">

            <label>
                🌿 Sepal Length
            </label>

            <input
                type="number"
                step="0.1"
                name="sepal_length"
                placeholder="Example: 5.1"
                required
            >

        </div>


        <div class="input-group">

            <label>
                🌿 Sepal Width
            </label>

            <input
                type="number"
                step="0.1"
                name="sepal_width"
                placeholder="Example: 3.5"
                required
            >

        </div>


        <div class="input-group">

            <label>
                🌺 Petal Length
            </label>

            <input
                type="number"
                step="0.1"
                name="petal_length"
                placeholder="Example: 1.4"
                required
            >

        </div>


        <div class="input-group">

            <label>
                🌺 Petal Width
            </label>

            <input
                type="number"
                step="0.1"
                name="petal_width"
                placeholder="Example: 0.2"
                required
            >

        </div>


        <button type="submit">

            🔮 Predict Flower

        </button>


    </form>


    {% if result %}

        <div class="result">

            🌸 Prediction: {{ result }}

        </div>

    {% endif %}


    <div class="footer">

        Powered by Python • Flask • Scikit-learn

    </div>


</div>


</body>

</html>
"""


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE
    )


# =========================================================
# REST API
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Receive JSON data
    data = request.get_json()

    # Extract input features
    features = data["features"]

    # Make prediction
    prediction = model.predict(
        [features]
    )

    # Convert prediction to integer
    predicted_class = int(
        prediction[0]
    )

    # Return JSON response
    return jsonify({

        "prediction": predicted_class,

        "class_name":
            class_names[
                predicted_class
            ]

    })

# =========================================================
# WEB PAGE PREDICTION
# =========================================================

@app.route(
    "/predict-web",
    methods=["POST"]
)
def predict_web():

    # Get values from HTML form

    sepal_length = float(
        request.form[
            "sepal_length"
        ]
    )

    sepal_width = float(
        request.form[
            "sepal_width"
        ]
    )

    petal_length = float(
        request.form[
            "petal_length"
        ]
    )

    petal_width = float(
        request.form[
            "petal_width"
        ]
    )


    # Create feature list

    features = [

        sepal_length,

        sepal_width,

        petal_length,

        petal_width

    ]

    # Predict flower class

    prediction = model.predict(
        [features]
    )

    # Convert prediction to integer

    predicted_class = int(
        prediction[0]
    )


    # Get flower name

    result = class_names[
        predicted_class
    ]


    # Display result on webpage

    return render_template_string(

        HTML_PAGE,

        result=result

    )


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000

    )