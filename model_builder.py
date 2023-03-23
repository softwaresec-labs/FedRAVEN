import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils


def generate_pickle():

    apks_combined_processed_csv = "LVDAndro_APKs_Combined_Processed_V3.csv"
    vulnerability_df = pd.read_csv(apks_combined_processed_csv, low_memory=False).fillna("")

    vulnerability_df = vulnerability_df[['Code', 'processed_code', 'Vulnerability_status', 'CWE_ID']]

    vulnerability_df.to_pickle("LVDAndro_APKs_Combined_Processed_V3.pickle")


def load_dataset():
    vulnerability_df = pd.read_pickle("LVDAndro_APKs_Combined_Processed_V3.pickle")
    return vulnerability_df


def process_dataset_binary(vulnerability_df):
    c_0 = vulnerability_df[vulnerability_df.Vulnerability_status == 0]
    c_1 = vulnerability_df[vulnerability_df.Vulnerability_status == 1]

    c_0_count = c_0.processed_code.count()
    c_1_count = c_1.processed_code.count()

    min_count = 0

    if c_0_count <= c_1_count:
        min_count = c_0_count
    else:
        min_count = c_1_count

    i = (math.ceil(min_count / 1000) * 1000)-1000
    print(min_count, i)

    df_0 = c_0.sample(i)
    df_1 = c_1.sample(i)

    vulnerability_df_binary = pd.concat([df_0, df_1], ignore_index=True)

    return vulnerability_df_binary


def train_model_binary(vulnerability_df_binary):

    code_list = vulnerability_df_binary.processed_code.tolist()
    y = vulnerability_df_binary.Vulnerability_status

    sentences = code_list
    y = y.values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1)
    vectorizer = CountVectorizer(analyzer='word', lowercase=True, max_df=0.80, min_df=40, ngram_range=(1, 3))
    vectorizer.fit(sentences_train)
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)

    print(len(vectorizer.vocabulary_))

    print(x_train.shape, y_train.shape)

    model = Sequential()
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    model_history = model.fit(x=x_train, y=y_train, epochs=1000, callbacks=early_stopping,
                              validation_data=(x_test, y_test))

    model.summary()

    model_history.history.keys()

    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    plt.title("Model Accuracy")
    plt.ylabel('accuracy')
    plt.ylabel('val_accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.title("Model Loss")
    plt.ylabel('loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()

    print(y_train)

    prediction = (model.predict(x_test) > 0.5).astype("int32")

    print(prediction[:5])
    print(y_test[:5])

    my_accuracy = accuracy_score(y_test, prediction.round())
    print(my_accuracy)

    my_f1_score = f1_score(y_test, prediction.round())
    print(my_f1_score)

    cm = confusion_matrix(y_test, prediction.round())
    print(cm)
    print(classification_report(y_test, prediction.round()))
    sn.heatmap(cm, annot=True, fmt='g')

    model.save('binary_model.h5')

    with open("binary_model.pickle", 'wb') as fout:
        pickle.dump((vectorizer, model), fout)


def process_dataset_multi(vulnerability_df):
    vulnerability_df = vulnerability_df.loc[vulnerability_df['Vulnerability_status'] == 1]
    vulnerability_df = vulnerability_df[['processed_code', 'CWE_ID']]

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-327", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-919", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-927", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-250", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-295", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-79", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-649", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-926", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-330", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-299", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-297", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-502", "Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-599", "Other")

    df_79 = c_79 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-79']
    df_89 = c_89 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-89']
    df_200 = c_200 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-200']
    df_250 = c_250 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-250']
    df_276 = c_276 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-276']
    df_295 = c_295 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-295']
    df_297 = c_297 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-297']
    df_299 = c_299 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-299']
    df_312 = c_312 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-312']
    df_327 = c_327 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-327']
    df_330 = c_330 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-330']
    df_502 = c_502 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-502']
    df_532 = c_532 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-532']
    df_599 = c_599 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-599']
    df_649 = c_649 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-649']
    df_676 = c_676 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-676']
    df_749 = c_749 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-749']
    df_919 = c_919 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-919']
    df_921 = c_921 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-921']
    df_925 = c_925 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-925']
    df_926 = c_926 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-926']
    df_927 = c_927 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-927']
    df_939 = c_939 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-939']
    df_other = c_other = vulnerability_df[vulnerability_df.CWE_ID == 'Other']

    df_532 = c_532.sample(10000)
    df_312 = c_312.sample(8000)

    vulnerability_df = pd.concat(
        [df_79, df_89, df_200, df_250, df_276, df_295, df_297, df_299, df_312, df_327, df_330, df_502, df_532, df_599,
         df_649, df_676, df_749, df_919, df_921, df_925, df_926, df_927, df_939, df_other], ignore_index=True)

    counts = vulnerability_df.CWE_ID.value_counts()
    print(counts)

    return vulnerability_df


def train_model_multi(vulnerability_df_multi):
    code_list = vulnerability_df_multi.processed_code.tolist()
    y = vulnerability_df_multi.CWE_ID

    sentences = code_list
    y = y.values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_y)

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, dummy_y, test_size=0.20,
                                                                        random_state=1)
    vectorizer = CountVectorizer(analyzer='word', lowercase=True, max_df=0.80, min_df=10, ngram_range=(1, 3))
    vectorizer.fit(sentences_train)
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)

    print(len(vectorizer.vocabulary_))

    print(x_train.shape, y_train.shape)

    model = Sequential()  # ANN
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    customised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=customised_optimizer, metrics=['accuracy'])

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    model_history = model.fit(x=x_train, y=y_train, epochs=1000, callbacks=early_stopping,
                              validation_data=(x_test, y_test))

    model.summary()

    model_history.history.keys()

    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    plt.title("Model Accuracy")
    plt.ylabel('accuracy')
    plt.ylabel('val_accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.title("Model Loss")
    plt.ylabel('loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()

    decoded_y = encoder.inverse_transform(np.argmax(y_test, axis=1))
    print(decoded_y)

    print(decoded_y[:10])

    prediction = model.predict(x_test)

    decoded_prediction = encoder.inverse_transform(np.argmax(prediction, axis=1))
    print(decoded_prediction[:10])

    my_accuracy = accuracy_score(decoded_y, decoded_prediction)
    print(my_accuracy)

    my_f1_score = f1_score(decoded_y, decoded_prediction, average='macro')
    print(my_f1_score)

    cm = confusion_matrix(decoded_y, decoded_prediction)
    print(classification_report(decoded_y, decoded_prediction))
    sn.heatmap(cm, annot=True, fmt='g')

    model.save('multiclass_model.h5')

    with open("multiclass_model.pickle", 'wb') as fout:
        pickle.dump((vectorizer, model, encoder), fout)


def main():
    generate_pickle()
    vulnerability_df = load_dataset()

    vulnerability_df_binary = process_dataset_binary(vulnerability_df)
    train_model_binary(vulnerability_df_binary)

    vulnerability_df_multi = process_dataset_multi(vulnerability_df)
    train_model_multi(vulnerability_df_multi)


if __name__ == '__main__':
    main()
