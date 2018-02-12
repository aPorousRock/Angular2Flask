# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for the server application."""

import json
import logging
import traceback
from datetime import datetime
from flask import Response, request, jsonify, current_app
from gevent.wsgi import WSGIServer
from flask_jwt_simple import (
    JWTManager, jwt_required, create_jwt, get_jwt_identity, get_jwt
)

from .http_codes import Status
from .factory import create_app, create_user

import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle
import pandas as pd
from keras.models import load_model,Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
K.set_image_dim_ordering('th')

Imagemodel = Sequential()
Imagemodel.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
Imagemodel.add(Activation('relu'))
Imagemodel.add(Convolution2D(64, (3, 3)))
Imagemodel.add(Activation('relu'))
Imagemodel.add(MaxPooling2D(pool_size=(2, 2)))
Imagemodel.add(Dropout(0.25))

Imagemodel.add(Convolution2D(64,(3, 3), padding='same'))
Imagemodel.add(Activation('relu'))
Imagemodel.add(Convolution2D(64, 3, 3))
Imagemodel.add(Activation('relu'))
Imagemodel.add(MaxPooling2D(pool_size=(2, 2)))
Imagemodel.add(Dropout(0.25))

Imagemodel.add(Flatten())
Imagemodel.add(Dense(512))
Imagemodel.add(Activation('relu'))
Imagemodel.add(Dropout(0.5))
Imagemodel.add(Dense(9))
Imagemodel.add(Activation('sigmoid'))
Imagemodel.load_weights("/Users/ajinkya.parkar@ibm.com/Documents/deep/keras_multilabel/multilabel/weights.11-0.72365.hdf5")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
Imagemodel.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
from IPython.display import Image
import cv2

logger = logging.getLogger(__name__)
app = create_app()
jwt = JWTManager(app)
model=load_model('LSTM5000.h5')

mod = gensim.models.Word2Vec.load('/Users/ajinkya.parkar@ibm.com/Downloads/apnews_sg/word2vec.bin'); 
    
    


@app.before_first_request
def init():
    """Initialize the application with defaults."""
    create_user(app)


@jwt.jwt_data_loader
def add_claims_to_access_token(identity):
    """Explicitly set identity and claims for jwt."""
    if identity == 'admin':
        roles = 'admin'
    else:
        roles = 'peasant'

    now = datetime.utcnow()
    return {
        'exp': now + current_app.config['JWT_EXPIRES'],
        'iat': now,
        'nbf': now,
        'sub': identity,
        'roles': roles
    }


@app.route("/api/logout", methods=['POST'])
@jwt_required
def logout():
    """Logout the currently logged in user."""
    # TODO: handle this logout properly, very weird implementation.
    identity = get_jwt_identity()
    if not identity:
        return jsonify({"msg": "Token invalid"}), Status.HTTP_BAD_UNAUTHORIZED
    logger.info('Logged out user !!')
    return 'logged out successfully', Status.HTTP_OK_BASIC


@app.route('/api/login', methods=['POST'])
def login():
    """View function for login view."""
    logger.info('Logged in user')

    params = request.get_json()
    username = params.get('username', None)
    password = params.get('password', None)

    if not username:
        return jsonify({"msg": "Missing username parameter"}), Status.HTTP_BAD_REQUEST
    if not password:
        return jsonify({"msg": "Missing password parameter"}), Status.HTTP_BAD_REQUEST

    # TODO Check from DB here
    if username != 'admin' or password != 'admin':
        return jsonify({"msg": "Bad username or password"}), Status.HTTP_BAD_UNAUTHORIZED

    # Identity can be any data that is json serializable
    # TODO: rather than passing expiry time here explicitly, decode token on client side. But I'm lazy.
    ret = {'jwt': create_jwt(identity=username), 'exp': datetime.utcnow() + current_app.config['JWT_EXPIRES']}
    return jsonify(ret), 200


@app.route('/api/protected', methods=['POST'])
@jwt_required
def get_data():
    """Get dummy data returned from the server."""
    jwt_data = get_jwt()
    

    data = {'Heroes': ['Hero1', 'Hero2', 'Hero3']}
    json_response = json.dumps(data)
    return Response(json_response,
                    status=Status.HTTP_OK_BASIC,
                    mimetype='application/json')

@app.route('/api/chat', methods=['POST'])
def get_chat():
    """Get dummy data returned from the server."""
    jwt_data = get_jwt()
    params = request.get_json()
    myText = params.get('myText', None)
    print(myText)
    print(params)
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(myText)
    sentvec = [mod[w] for w in sent if w in mod.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(5)]
    output=' '.join(outputlist)
    print(output)
    
    data = {'Heroes': ['Hero1', 'Hero2', 'Hero3']}
    
    
    json_response = json.dumps(output)
    return Response(json_response,
                    status=Status.HTTP_OK_BASIC,
                    mimetype='application/json')

@app.route('/api/image', methods=['POST'])
def get_Image():
    """Get dummy data returned from the server."""
    jwt_data = get_jwt()
    params = request.get_json()
    myText = params.get('myText', None)
    print(myText)
    print(params)
    img = cv2.imread(myText)
    img = cv2.resize(img,(100,100))
    img = img.transpose((2,0,1))
    img = img.astype('float32')
    img = img/255
    img = np.expand_dims(img,axis=0)
    pred = Imagemodel.predict(img)
    y_pred = np.array([1 if pred[0,i]>=0.6 else 0 for i in range(pred.shape[1])])
    
    finalOutput = []
    for key, value in enumerate(y_pred):
    
        if key == 0 and value == 1:
            finalOutput.append("Good for lunch")
    
        if key == 1 and value == 1:
            finalOutput.append("Good for dinner")
    
        if key == 2 and value == 1:
            finalOutput.append("Takes reservation")
        
        if key == 3 and value == 1:
            finalOutput.append("Outdoor seating")
        if key == 4 and value == 1:
            finalOutput.append("Restaurent is expensive")
        if key == 5 and value == 1:
            finalOutput.append("Has alchohol")
        if key == 6 and value == 1:
            finalOutput.append("Has Table Service")
        if key == 7 and value == 1:
            finalOutput.append("Ambience is classy")
        if key == 8 and value == 1:
            finalOutput.append("Good for kids")
    print(finalOutput)
    data = {'Heroes': ['Hero1', 'Hero2', 'Hero3']}
    
    
    json_response = json.dumps(finalOutput)
    return Response(json_response,
                    status=Status.HTTP_OK_BASIC,
                    mimetype='application/json')


def main():
    """Main entry point of the app."""
    try:
        port = 8080
        ip = '0.0.0.0'
        http_server = WSGIServer((ip, port),
                                 app,
                                 log=logging,
                                 error_log=logging)
        print("Server started at: {0}:{1}".format(ip, port))
        http_server.serve_forever()
    except Exception as exc:
        logger.error(exc.message)
        logger.exception(traceback.format_exc())
    finally:
        # Do something here
        pass
