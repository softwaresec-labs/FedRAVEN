import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,f1_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import to_categorical



def train_model(vulnerability_df):

    print(vulnerability_df.CWE_ID.value_counts())

    # vulnerability_df = vulnerability_df[vulnerability_df.CWE_ID != ""]
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("","Other")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-327","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-919","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-927","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-250","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-295","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-79","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-649","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-926","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-330","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-299","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-297","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-502","Other")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-599","Other")

    print(vulnerability_df.CWE_ID.value_counts())

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

    # df_532 = c_532.sample(10000)
    # df_312 = c_312.sample(8000)

    vulnerability_df = pd.concat([df_79, df_89,df_200,df_250,df_276,df_295,df_297,df_299,df_312,df_327,df_330,df_502,df_532,df_599,df_649,df_676,df_749,df_919,df_921,df_925,df_926,df_927,df_939,df_other], ignore_index=True)

    counts = vulnerability_df.CWE_ID.value_counts()

    print(counts)

    code_list = vulnerability_df.processed_code.tolist()
    y = vulnerability_df.CWE_ID

    sentences = code_list
    y = y.values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_Y)

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, dummy_y, test_size=0.20, random_state=0)
    vectorizer = CountVectorizer(analyzer='word', lowercase=True, max_df=0.80, min_df=10, ngram_range=(1, 3),max_features=300)
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train).toarray()
    X_test = vectorizer.transform(sentences_test).toarray()

    print(len(vectorizer.vocabulary_))

    print(X_train.shape, y_train.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense((y_train.shape)[1], activation='softmax'))
    # model.add(Dense((y_train.shape)[1], activation='softmax'))  # Output Layer

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

    return model, early_stopping, X_train, y_train, X_test, y_test