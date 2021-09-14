from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import  pandas as pd

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    ###########standarad scaling
    train_data_path = config["split_data"]["train_path"]
    train = pd.read_csv(train_data_path, sep=",")
    drop_col1 = config["base"]["drop_col1"]
    drop_col2 = config["base"]["drop_col2"]
    drop_col3 = config["base"]["drop_col3"]
    drop_col4 = config["base"]["drop_col4"]
    drop_col5 = config["base"]["drop_col5"]
    drop_col = [drop_col1, drop_col2, drop_col3, drop_col4, drop_col5]
    train = pd.read_csv(train_data_path, sep=",")
    X_train_H = train.drop(columns=drop_col, axis=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_H)

    ###################prediction############
    data_pre = scaler.transform(data)
    prediction = model.predict(data_pre)
    print(prediction)
    return prediction



@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                data = dict(request.form).values()
                data1 = [list(map(float,data))]

                response = predict(data1)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)