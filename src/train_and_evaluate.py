# split the raw data
# save it in data/processed folder
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from get_data import read_params
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
import pickle

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alphaE = config["estimators"]["ElasticNet"]["params1"]["alpha"]
    #l1_ratio = config["estimators"]["ElasticNet"]["params1"]["l1_ratio"]
    cvE = config["estimators"]["ElasticNet"]["params1"]["cv"]

    alphaA = config["estimators2"]["RidgeCV"]["params2"]["alphaA"]
    alphaB = config["estimators2"]["RidgeCV"]["params2"]["alphaB"]
    alphaC = config["estimators2"]["RidgeCV"]["params2"]["alphaC"]
    cvR = config["estimators2"]["RidgeCV"]["params2"]["cvR"]

    maxiterL = config["estimators3"]["LassoCV"]["params3"]["maxiter"]
    cvL = config["estimators3"]["LassoCV"]["params3"]["cv"]

    target = [config["base"]["target_col"]]
    drop_col1 = config["base"]["drop_col1"]
    drop_col2 = config["base"]["drop_col2"]
    drop_col3 = config["base"]["drop_col3"]
    drop_col4 = config["base"]["drop_col4"]
    drop_col5 = config["base"]["drop_col5"]
    drop_col = [drop_col1,drop_col2,drop_col3,drop_col4,drop_col5]
    #project_root = os.path.dirname(os.path.dirname(__file__))
    #train = pd.read_csv(os.path.join(project_root,train_data_path), sep=",")
    train = pd.read_csv(train_data_path, sep=",")
    print(train.columns)
    Y_train = train[target]
    #for i in drop_col:
    #    X_train = train.drop(columns=[i],axis=1)


    X_train_H = train.drop(columns=drop_col,axis=1)
    print(X_train_H.columns)

    pickle.dump(X_train_H, open('train_data.pkl', 'wb'))
    pickle.load(open('train_data.pkl', 'rb'))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_H)


    ##print(X_train.head(5))

    #test = pd.read_csv(os.path.join(project_root,test_data_path), sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    Y_test = test[target]
    X_test_H = test.drop(columns=drop_col,axis=1)
    print(X_test_H.columns)
    X_test = scaler.transform(X_test_H)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    r2 = regressor.score(X_test, Y_test)

    from sklearn.linear_model import Ridge, RidgeCV
    ridgecv = RidgeCV(alphas=(alphaA,alphaB,alphaC), cv=cvR, normalize=True)
    ridgecv.fit(X_train, Y_train)
    ridge_lr = Ridge(alpha=ridgecv.alpha_)
    ridge_lr.fit(X_train, Y_train)
    ridge_score = ridge_lr.score(X_test, Y_test)
    print('ridge_score:', ridge_score)


    from sklearn.linear_model import Lasso, LassoCV
    lassocv = LassoCV(cv=cvL, max_iter=maxiterL, normalize=True)
    lassocv.fit(X_train, Y_train)
    lasso = Lasso(alpha=lassocv.alpha_)
    lasso.fit(X_train, Y_train)
    lass_score = lasso.score(X_test, Y_test)
    print('lass_score:', lass_score)

    from sklearn.linear_model import ElasticNet, ElasticNetCV
    elastic = ElasticNetCV(alphas=None, cv=cvE)
    elastic.fit(X_train, Y_train)
    elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio)
    elastic_lr.fit(X_train, Y_train)
    elastic_score = elastic_lr.score(X_test, Y_test)
    print('elastic_score:', elastic_score)
    model_all = {'R2 Linear': r2, 'ridge_score': ridge_score, 'lass_score': lass_score,
                 'elastic_score': elastic_score}
    Keymax = max(model_all, key=lambda x: model_all[x])

    if Keymax == 'R2 Linear':
        model_selection = regressor
    if Keymax == 'ridge_score':
        model_selection = ridge_lr
    if Keymax == 'lass_score':
        model_selection = lasso
    if Keymax == 'elastic_score':
        model_selection = elastic_lr

    predicted_qualities = model_selection.predict(X_test)
    print(predicted_qualities)
    (rmse, mae, r2) = eval_metrics(Y_test, predicted_qualities)


        #####################################################


    save_dir = config["model_dir"]
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    #saved_dir_con = os.path.join(project_root, save_dir)
    #print(saved_dir_con)
    #os.getcwd()
    #os.chdir(saved_dir_con)
    #os.makedirs("report", exist_ok=True)

    #with open(os.path.join(saved_dir_con, scores_file), "w") as f:
    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    #with open(os.path.join(saved_dir_con, params_file), "w") as f:
    with open(params_file, "w") as f:
        params = {
            "alpha": alphaE,
            #"l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)
    #####################################################

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(model_selection, model_path)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)