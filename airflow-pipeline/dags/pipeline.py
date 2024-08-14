import os
import pandas as pd
import numpy as np
from datetime import datetime
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, log_loss


def prepare_data():
    import pandas as pd
    print("---- inside prepare_data component ------")
    df= pd.read_csv("data/my_paypal_creditcard (1).csv")
    df=df.dropna()
    df=df.drop_duplicates()
    df.to_csv(f"data/cleaned_df.csv", index=False)

def handling_imbalance():
    import pandas as pd
    from sklearn.utils import resample
    print("-----inside imbalance_handling component ------")
    
    df_new=pd.read_csv("data/cleaned.csv")

    # downsampling the majority class to have same rows as the minority class
    downsampled_df=resample(df_new[df_new['Class']==0],
                            n_samples=len(df_new[df_new['Class']==1]),
                            random_state=42)

    # merging minority and new downsampled majority
    df_balanced= pd.concat([df_new[df_new['Class']==1], downsampled_df])
    df_balanced.to_csv(f"data/balanced_df.csv", index=False)


def train_test_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    #load data
    df_balanced=pd.read_csv("data/balanced_df.csv")

    #data splitting into test and train sets
    X= df_balanced.drop('Class', axis=1)
    Y= df_balanced['Class']
    x_train, x_test,y_train, y_test= train_test_split(X, Y, random_state=42, test_size=0.3)

    # Feature Scaling of features
    sc= StandardScaler()
    x_train= sc.fit_transform(x_train)
    x_test= sc.transform(x_test)
    y_train=y_train.to_numpy().flatten()
    y_test=y_test.to_numpy().flatten()

    #save as numpy objects
    np.save(f"x_train.npy", x_train)
    np.save(f"y_train.npy", y_train)
    np.save(f"x_test.npy", x_test)
    np.save(f"y_test.npy", y_test)


def model_training():

    print("-----inside training component-----")
    import pickle
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    #load train and test splits
    x_train= np.load(f"x_train.npy", allow_pickle=True)
    y_train= np.load(f"y_train.npy", allow_pickle=True)

    best_params={
            "C":2.6925812675005436,
            'max_iter':365,
            'random_state':42,
            'fit_intercept':True,
            'solver':'liblinear',
            'warm_start':False
            }
    
    lr= LogisticRegression(**best_params)
    lr.fit(x_train, y_train)
    
    filename = 'models/logistic-credit-fraud-model.bin'
    pickle.dump(lr, open(filename, 'wb'))


def predict_test_data():
    print("-----inside prediction component-----")
    import pickle
    import numpy as np
    
    #load  test data
    x_test= np.load(f"x_test.npy", allow_pickle=True)
    with open(f'models/logistic-credit-fraud-model.bin', 'wb') as f:
        model= pickle.load(f)
    y_pred = model.predict(x_test)
    np.save(f"y_pred.npy", y_pred)


def evaluation_metrics():
    print("-----inside evaluation metrics component-----")
    import numpy as np
    from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, log_loss
    from sklearn import metrics

    #load prediction and actual values
    y_test= np.load(f"y_test.npy", allow_pickle=True)
    y_pred= np.load(f"y_pred.npy", allow_pickle=True)

    acc=accuracy_score(y_test, y_pred)
    loss=log_loss(y_test, y_pred)
    pre=precision_score(y_test, y_pred, average='micro')
    recall=recall_score(y_test, y_pred, average='micro')
    f1= f1_score(y_test, y_pred, average='micro')
    print(metrics.classification_report(y_test, y_pred))
    
    print("/n Metrics:", {"accuracy": round(acc,3), "precision": round(pre,3), "recall": round(recall,3), "f1 score": round(f1,3) ,"log loss": round(loss,3)})


'''
dag1= DAG(
    dag_id= "ml_pipeline_demo",
    schedule_interval=  "@daily",
    start_date= datetime(2024,8,11),
    catchup=False
)
'''

with DAG(
    dag_id= "ml_pipeline_demo",
    schedule_interval=  "@daily",
    start_date= datetime(2024,8,11),
    catchup=False
) as dag:

    prepare_data= PythonOperator(
        task_id= "prepare_data",
        python_callable= prepare_data,
    )

    handling_imbalance= PythonOperator(
        task_id= "handling_imbalance",
        python_callable= handling_imbalance,
    )
    
    train_test_split= PythonOperator(
        task_id= "train_test_split",
        python_callable= train_test_split,
    )
    
    model_training= PythonOperator(
        task_id= "model_training",
        python_callable= model_training,
    )
    
    predict_test_data= PythonOperator(
        task_id= "predict_test_data",
        python_callable= predict_test_data,
    )
    
    evaluation= PythonOperator(
        task_id= "evaluation_metrics",
        python_callable= evaluation_metrics,
    )

    prepare_data >> handling_imbalance >> train_test_split >> model_training >> predict_test_data >> evaluation

    