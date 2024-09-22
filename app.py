from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route Home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            job=request.form.get('job'),
            education=request.form.get('education'),
            contact=request.form.get('contact'),
            poutcome=request.form.get('poutcome'),
            age_group=request.form.get('age_group'),
            duration=request.form.get('duration'),
            campaign=request.form.get('campaign'),
            pdays=request.form.get('pdays'),
            previous=request.form.get('previous'),
            emp_var_rate=request.form.get('emp_var_rate'),
            cons_price_idx=request.form.get('cons_price_idx'),
            cons_conf_idx=request.form.get('cons_conf_idx')
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        

        return render_template('home.html', results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)