from flask import Flask, request, render_template
import pandas as pd
import joblib

#Declare Flask App
app = Flask(__name__)

#@app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":
        # Get values through input bars
        tweet = request.form.get("tweet")

        #prediction = clf.predict(X)[0]
        prediction = tweet
    else:
        prediction = "No Tweet"

    return render_template("index.html", output = prediction)
#Run the App
if __name__ == "__main__":
    app.run(debug = True)
