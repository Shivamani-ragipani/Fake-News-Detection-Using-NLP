from flask import Flask, request, render_template, url_for
import pickle

# Load the vectorizer and model
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

app = Flask(__name__)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
    app.run(debug=True, host='0.0.0.0', port=port)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)

        # Predict the label for the given news headline
        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        # Determine which GIF to display based on prediction
        gif_path = "success.gif" if predict == "REAL" else "fail.gif"

        return render_template(
            "prediction.html",
            prediction_text="News headline is classified as -> {}".format(predict),
            gif_path=url_for('static', filename=gif_path)  # Use url_for to resolve the static file path
        )
    else:
        return render_template("prediction.html", prediction_text="Enter a news headline.", gif_path=None)

if __name__ == '__main__':
    app.debug = True
    app.run()
