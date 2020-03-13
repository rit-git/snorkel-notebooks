from sklearn.feature_extraction.text import CountVectorizer
from snorkel.analysis import metric_score
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import preds_to_probs
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


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


def get_keras_bilstm(vocab_size = 20000, output_dim=2, maxlen=100):
    set_seeds()
    
    # cut texts after this number of words
    # (among top max_features most common words)

    model = Sequential()
    model.add(Embedding(maxlen, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.math.softmax
    dense = tf.keras.layers.Dense(
        units=output_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )
    model.add(dense)
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


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


def train_model(label_model, df_train, df_valid, df_test, L_train):
    # Train a model on top

    probs_train = label_model.predict_proba(L=L_train)

    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )
    print("{} out of {} examples used for training data".format(len(df_train_filtered), len(df_train)))

    return train_model_from_probs(df_train_filtered, probs_train_filtered, df_valid, df_test)

def train_model_from_probs(df_train_filtered, probs_train_filtered, df_valid, df_test):
    set_seeds()
    
    vectorizer = CountVectorizer(ngram_range=(1, 2))
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

    preds_test = keras_model.predict(x=X_test).argmax(axis=1)
    test_acc = metric_score(golds=Y_test, preds=preds_test, metric="accuracy")
    print(f"Test Accuracy: {test_acc * 100:.1f}%")
    test_f1 = metric_score(golds=Y_test, preds=preds_test, metric="f1")
    print(f"Test F1: {test_f1 * 100:.1f}%")
    test_prec = metric_score(golds=Y_test, preds=preds_test, metric="precision")
    print(f"Test Precision: {test_f1 * 100:.1f}%")
    test_recall = metric_score(golds=Y_test, preds=preds_test, metric="recall")
    print(f"Test Recall: {test_f1 * 100:.1f}%")

    return {
        "test_acc": test_acc, 
        "test_f1": test_f1, 
        "test_precision": test_prec, 
        "test_recall": test_recall,
        "model": "logreg"
    }



