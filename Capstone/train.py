from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.workspace import Workspace
from azureml.core import Datastore

def cleandata(data):
    alpha = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8, "I":9, "J":10, "K":11, "L":12, "M":13, "N":14, "O":15, "P":16, "Q":17, "R":18, "S":19, "T":20,"U":21, "V":22, "W":23, "X":24, "Y":25, "Z":26}
    x_df = data.to_pandas_dataframe().head(30000).dropna()
    x_df.drop("phone_number", inplace=True, axis=1)
    x_df.drop("applicant_name", inplace=True, axis=1)
    x_df.drop("billing_address", inplace=True, axis=1)
    x_df.drop("billing_city", inplace=True, axis=1)
    x_df.drop("EVENT_TIMESTAMP", inplace=True, axis=1)
    ip_addresses = pd.get_dummies(x_df.ip_address, prefix="ip_address")
    x_df.drop("ip_address", inplace=True, axis=1)
    x_df = x_df.join(ip_addresses)
    user_agents = pd.get_dummies(x_df.user_agent, prefix="user_agent")
    x_df.drop("user_agent", inplace=True, axis=1)
    x_df = x_df.join(user_agents)
    email_domains = pd.get_dummies(x_df.email_domain, prefix="email_domain")
    x_df.drop("email_domain", inplace=True, axis=1)
    x_df = x_df.join(email_domains)
    billing_states = pd.get_dummies(x_df.billing_state, prefix="billing_state")
    x_df.drop("billing_state", inplace=True, axis=1)
    x_df = x_df.join(billing_states)
    currencies = pd.get_dummies(x_df.currency, prefix="currency")
    x_df.drop("currency", inplace=True, axis=1)
    x_df = x_df.join(currencies)
    merchant_ids = pd.get_dummies(x_df.merchant_id, prefix="merchant_id")
    x_df.drop("merchant_id", inplace=True, axis=1)
    x_df = x_df.join(merchant_ids)
    locales = pd.get_dummies(x_df.locale, prefix="locale")
    x_df.drop("locale", inplace=True, axis=1)
    x_df = x_df.join(locales)
    x_df["cvv"] = x_df.cvv.map(alpha)
    x_df["signature_image"] = x_df.signature_image.map(alpha)
    x_df["transaction_type"] = x_df.transaction_type.map(alpha)
    x_df["transaction_env"] = x_df.transaction_env.map(alpha)
    x_df["tranaction_initiate"] = x_df.tranaction_initiate.map(alpha)
    y_df = x_df.pop("EVENT_LABEL").apply(lambda s: 1 if s == "legit" else 0)
    return x_df, y_df



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
        
    ws = run.experiment.workspace
    defaultDS = Datastore(ws,name='workspaceblobstore')
    ds=TabularDatasetFactory.from_delimited_files(path=[(defaultDS,'**/fraud_challenge_150k.csv')])
    
    x, y = cleandata(ds)

    # Split data into train and test sets

    ### YOUR CODE HERE ###a
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    ##model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')
    ##print(accuracy)

if __name__ == '__main__':
    main()