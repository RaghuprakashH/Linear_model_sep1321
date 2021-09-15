from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import  pandas as pd
from  cassandra.query import SimpleStatement,BatchStatement
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

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

    X_train_H = pd.read_pickle('train_data.pkl')
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
                cloud_config = {
                    'secure_connect_bundle': 'secure-connect-test1.zip'
                }

                auth_provider = PlainTextAuthProvider('DzgSXTSzQgWNpYocAWPpQzAX',
                                                      '27C9coC--crqmF0MiZldjv9Kg8NyhTzMP66SOPbHtiaNOWcidhyBz1FuOIuUp.,p2CajK266pu2QEhLkCNs4Zkt6qQaSce2cS_+10a9clpH6UhkdUkNtuBoTczw8sK_X')

                cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
                session = cluster.connect()
                data = dict(request.form).values()
                data1 = [list(map(float,data))]
                HDF = data1[0][5]
                OSF =data1[0][7]

                Process_temperature = data1[0][0]
                PWF = data1[0][6]
                RNF = data1[0][8]
                Rotational_speed = data1[0][1]
                Tool_wear = data1[0][3]
                Torque = data1[0][2]
                TWF = data1[0][4]
                countrec = session.execute("SELECT COUNT(*) FROM ineuron3.AirTempPredict;").one()
                cno1 = countrec[0]
                cno = cno1 + 1
                response = predict(data1)
                prediction = response[0]
                final_data = [(cno,HDF,OSF,prediction,Process_temperature,PWF,RNF,Rotational_speed,Tool_wear,Torque,TWF)]

                qr1 = 'INSERT INTO ineuron3.AirTempPredict(cno,hdf,osf,prediction,process_temperature,pwf,rnf,rotational_speed,tool_wear,torque,twf) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                batch = BatchStatement()
                for i in final_data:
                    final_data1 = i
                batch.add(qr1, final_data1)
                session.execute(batch)
                return render_template("index.html", response=response[0])
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)