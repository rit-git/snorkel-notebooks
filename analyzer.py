import copy
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from data.preparer import *
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient import discovery
from modeler import Modeler
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.analysis import metric_score
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import preds_to_probs
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

stat_history = pd.DataFrame()
modeler = None

def load_dataset(task: str, DELIMITER='#'):
    set_seeds()
    if task == "Amazon":
        df_train, df_dev, df_valid, df_test, df_test_heldout = load_amazon_dataset(delimiter=DELIMITER)

    elif task =="Youtube":
        df_train, df_dev, df_valid, df_test, df_test_heldout = load_youtube_dataset(delimiter=DELIMITER)

    elif task == "Film":
        df_train, df_dev, df_valid, df_test, df_test_heldout = load_film_dataset()

    elif (task == "News") or (task == "Debug"):
        df_train, df_dev, df_valid, df_test, df_test_heldout = load_news_dataset()

    global modeler
    modeler = Modeler(df_train, df_dev, df_valid, df_test, df_test_heldout)
    update_stats({}, "begin")

    return (df_train, df_dev, df_valid, df_test)

def set_seeds():
    # set all random seeds
    import tensorflow as tf
    from numpy.random import seed as np_seed
    from random import seed as py_seed
    from snorkel.utils import set_seed as snork_seed
    snork_seed(123)
    tf.random.set_seed(123)
    np_seed(123)
    py_seed(123)


def get_keras_logreg(input_dim, output_dim=2):
    set_seeds()

    model = Sequential()
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.math.softmax
    dense = tf.keras.layers.Dense(
        units=output_dim,
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )
    model.add(dense)
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

def get_keras_early_stopping(patience=10, monitor="val_accuracy"):
    """Stops training if monitor value doesn't exceed the current max value after patience num of epochs"""
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=1, restore_best_weights=True
    )


def train_model(label_model, L_train):
    probs_train = label_model.predict_proba(L=L_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=modeler.df_train, y=probs_train, L=L_train
    )
    print("{} out of {} examples used for training data".format(len(df_train_filtered), len(modeler.df_train)))

    return train_model_from_probs(df_train_filtered, probs_train_filtered, modeler.df_valid, modeler.df_test)

def train_model_from_probs(df_train_filtered, probs_train_filtered, df_valid, df_test):
    set_seeds()
    
    vectorizer = modeler.vectorizer
    X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())

    X_valid = vectorizer.transform(df_valid["text"].tolist())
    X_test = vectorizer.transform(df_test["text"].tolist())

    Y_valid = df_valid["label"].values
    Y_test = df_test["label"].values

    # Define a vanilla logistic regression model with Keras
    keras_model = get_keras_logreg(input_dim=X_train.shape[1])

    keras_model.fit(
        x=X_train,
        y=probs_train_filtered,
        validation_data=(X_valid, preds_to_probs(Y_valid, 2)),
        callbacks=[get_keras_early_stopping()],
        epochs=50,
        verbose=0,
    )

    modeler.keras_model = keras_model

    preds_test = keras_model.predict(x=X_test).argmax(axis=1)

    stats = modeler.get_stats(modeler.Y_test, preds_test)

    update_stats({**stats, "data": "test"}, "train_model")

    return stats

def update_stats(new_stats_dict: dict, action: str, label_model=None, applier=None):
    if applier is not None:
        modeler.L_heldout = applier.apply(df=modeler.df_heldout)
    if label_model is not None:
        modeler.label_model = label_model
    global stat_history
    new_stats_dict = copy.deepcopy(new_stats_dict)
    
    new_stats_dict.update({
        "time": datetime.now(), 
        "action": action
    })
    stat_history = stat_history.append(new_stats_dict, ignore_index=True)

    if action=="train_model":
        heldout_stats = heldout_stats = modeler.get_heldout_lr_stats()
        if len(heldout_stats) > 0:
            stat_history = stat_history.append({
                "action": "heldout_test_LR_stats",
                "time": datetime.now(),
                "data": "heldout",
                **heldout_stats
            }, ignore_index=True)
    elif (action=="stats"):
        heldout_stats = modeler.get_heldout_stats()
        if len(heldout_stats) > 0:
            stat_history = stat_history.append({
                "action": "heldout_test_stats",
                "time": datetime.now(), 
                "data": "heldout",
                **heldout_stats
            }, ignore_index=True)



def save_model(dirname):
    update_stats({"dirname": dirname}, "save_model")
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    modeler.save(dirname)

    global stat_history
    stat_history["time_delta"] = stat_history["time"] - stat_history["time"].iloc[0]
    stat_history.to_csv(os.path.join(dirname, "statistics_history.csv"))

def upload_stats(stats_file, file_name): 
    dir_path = os.path.dirname(os.path.realpath(__file__))

    GOOGLE_APPLICATION_CREDENTIALS=os.path.join(dir_path, "data/credentials.json")
    creds = credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=['https://www.googleapis.com/auth/drive'])
    drive_api = discovery.build('drive', 'v3', credentials=creds)
    drive_client = drive_api.files()

    file_metadata = {'name': file_name, 'parents':["1bYXU5TwT_jvmuygkBbBy2r-BN7JUBHX5"]}
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(stats_file,
                            mimetype='text/csv')

    create_kwargs = {
        'body': file_metadata,
        'media_body': media,
        'fields': 'id'
    }

    file = drive_client.create(**create_kwargs).execute()

def upload_data(zipfile):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    GOOGLE_APPLICATION_CREDENTIALS=os.path.join(dir_path, "data/credentials.json")
    creds = credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=['https://www.googleapis.com/auth/drive'])
    drive_api = discovery.build('drive', 'v3', credentials=creds)
    drive_client = drive_api.files()

    file_metadata = {'name': zipfile, 'parents':["1bYXU5TwT_jvmuygkBbBy2r-BN7JUBHX5"]}
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(zipfile,
                            mimetype='application/zip')

    create_kwargs = {
        'body': file_metadata,
        'media_body': media,
        'fields': 'id'
    }

    file = drive_client.create(**create_kwargs).execute()
    print( 'File ID: '  +  file.get('id'))