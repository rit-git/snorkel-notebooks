from sklearn.feature_extraction.text import CountVectorizer
from snorkel.analysis import metric_score
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import preds_to_probs
import tensorflow as tf


def get_keras_logreg(input_dim, output_dim=2):
    model = tf.keras.Sequential()
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.nn.softmax
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

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())

    X_valid = vectorizer.transform(df_valid.text.tolist())
    X_test = vectorizer.transform(df_test.text.tolist())

    Y_valid = df_valid.label.values
    Y_test = df_test.label.values

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