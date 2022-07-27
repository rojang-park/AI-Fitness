import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
import seaborn as sns
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import keras
import random
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

Xtrain = pd.read_csv('train_df.csv')
Xtest = pd.read_csv('test_df.csv')
Ytrain = pd.read_csv( 'labels.csv', index_col='id')
submission = pd.read_csv('sample_submission.csv', index_col='id')
print(Ytrain.key.value_counts())

Ytrain[['key','exercise','description']] = Ytrain[['key','exercise','description']].astype('category')

feature_names = ['nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x',
       'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',
       'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x',
       'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x',
       'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
       'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x',
       'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x',
       'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x',
       'right_ankle_y']

# scaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()

Xtrain_scaled = pd.DataFrame(data=scaler.fit_transform(Xtrain[feature_names]), index=Xtrain.index, columns=feature_names)
Xtest_scaled = pd.DataFrame(data=scaler.transform(Xtest[feature_names]), index=Xtest.index, columns=feature_names)

print(Xtrain_scaled.shape, Xtest_scaled.shape)

#reshape
Xtrain_scaled = np.array(Xtrain_scaled).reshape(-1, 31, len(feature_names)).astype('float32')
Xtest_scaled= np.array(Xtest_scaled).reshape(-1, 31, len(feature_names)).astype('float32')

Ytrain = Ytrain['key']

print(Xtrain_scaled.shape, Ytrain.shape, Xtest_scaled.shape, submission.shape)

#modeling

# 데이터를 하나하나마다 다른 Rolling 과 다른 노이즈를 추가하여 오버샘플링 하는 용도의 함수
def aug_data(w, noise=True, roll_max=550, roll_min=50, noise_std=0.02):
    assert w.ndim == 3
    auged=[]

    for i in range(w.shape[0]):
        roll_amount = np.random.randint(roll_min, roll_max)
        data = np.roll(w[i:i+1], shift=roll_amount, axis=1)
        if noise:
            gaussian_noise = np.random.normal(0, noise_std, data.shape)
            data += gaussian_noise
        auged.append(data)
    
    auged = np.concatenate(auged)
    return auged


# 모델의 인풋 바로 다음에 랜덤한 값으로 Rolling 을 하는 커스텀 레이어. 
class Rolling(Layer):
    def __init__(self, roll_max=599, roll_min=0):
        super(Rolling, self).__init__()
        self.random_roll = random.randint(roll_min, roll_max)   
        
    #def build(self, input_shape):  # Create the state of the layer (weights)
    #    pass
    
    def call(self, inputs, training=None):# Defines the computation from inputs to outputs
        if training:
            return tf.roll(inputs, shift=self.random_roll, axis=1)
        else:
            return inputs
        
    def get_config(self):
        return {'random_roll': self.random_roll}


#GPU 메모리가 생각보다 많이 안필요한것 같아서, 메모리 60%로 제한. 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Convolution, Dense 레이어 여러번 쓰기 번거로우니까 만든 함수
def ConvBlock3(w, kernel_size, filter_size, activation):
    x_res = Conv1D(filter_size, kernel_size, kernel_initializer='he_uniform', padding='same')(w)
    x = BatchNormalization()(x_res)
    x = Activation(activation)(x)
    x = Conv1D(filter_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv1D(filter_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    x = Add()([x, x_res])
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def DenseBNAct(w, dense_units, activation):
    x = Dense(dense_units, kernel_initializer='he_uniform')(w)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x



# CV 해서 train 데이터의 cross_val_predict 와 test 의 Averaged Prediction 을 리턴할 수 있도록 만듬. 
def cross_validate(build_fn, build_fn_params={}, X=Xtrain_scaled, Y=Ytrain, Xt=Xtest_scaled, batch_size=16, random_state=None,
                   num_folds=10, stratify=True, 
                   predict_train=True, predict_test=True, verbose=False, 
                   aug=False, aug_noise=True, aug_noise_std=0.03, num_augs=1, 
                   save=False, save_filename_prefix=None):
    if save:
        assert save_filename_prefix != None
        
    if stratify:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    
    if predict_train:
        train_pred = np.zeros(shape=(X.shape[0], 64))
    
    score = 0
    num_fold = 1
    if predict_test:
        model_list=[]
        test_pred = np.zeros(shape=submission.shape)
        
    for train_ind, valid_ind in tqdm(kf.split(X, Y)):
        x_train = X[train_ind]
        y_train = Y[train_ind]
        x_valid = X[valid_ind]
        y_valid = Y[valid_ind]
        
        if aug:
            aug_ind = y_train.loc[y_train!=26].index
            for _ in range(num_augs):
                aug_x = aug_data(X[aug_ind], noise=aug_noise, noise_std=aug_noise_std)
                aug_y = Y[aug_ind]
                x_train = np.concatenate([x_train, aug_x], axis=0)
                y_train = pd.concat((y_train, aug_y), axis=0)
                assert x_train.shape[0] == y_train.shape[0]

        model = build_fn(**build_fn_params)
        reduce_lr = ReduceLROnPlateau(mode='min', monitor='val_loss', verbose=verbose, factor=0.3, patience=25)
        es = EarlyStopping(mode='min', monitor='val_loss', restore_best_weights=True, verbose=verbose, patience=76)
        fit_history = model.fit(x_train, y_train, epochs=5000, shuffle=True, verbose=verbose, 
                                callbacks=[es, reduce_lr], batch_size=batch_size, validation_data=(x_valid, y_valid)
                               )
        score += min(fit_history.history['val_loss'])
        
        if verbose:
            print(f"Fold {num_fold} Score : {min(fit_history.history['val_loss'])}")
        if save:
            model.save(f"{save_filename_prefix} - Fold{num_fold} Model.h5")
        
        
        if predict_train:
            train_pred[valid_ind] = model.predict(x_valid)
        if predict_test:
            model_list.append(model)
            
        num_fold += 1

    print(f"Score : {score / num_folds}")
    return_list=[]
    if predict_train:
        train_pred = pd.DataFrame(data=train_pred, index=Y.index, columns=submission.columns)
        return_list.append(train_pred)
        
    if predict_test:
        for model in tqdm(model_list):
            test_pred += model.predict(Xt)
        
        return_list.append(pd.DataFrame(data=test_pred/num_folds, index=submission.index, columns=submission.columns))
        
    return_list.append(score / num_folds)
        
    return return_list

    def build_fn(lr = 0.001):
    activation='elu'
    kernel_size=9
    
    
    model_in = Input(shape=Xtrain_scaled.shape[1:])
    x = Rolling(roll_max=599, roll_min=0)(model_in)
    x = SpatialDropout1D(0.1)(x)
    
    ##################################### FIX #####################################
    x = ConvBlock3(x, kernel_size=kernel_size, filter_size=128, activation=activation)
    x = MaxPooling1D(3)(x)
    x = SpatialDropout1D(0.1)(x)
    
    
    x = ConvBlock3(x, kernel_size=kernel_size, filter_size=128, activation=activation)
    x = GlobalAveragePooling1D()(x)
    ###############################################################################
    
    x = DenseBNAct(x, dense_units=64, activation=activation)
    x = Dropout(0.4)(x)
    
    
    model_out = Dense(units=64, activation='softmax')(x)
    model = Model(model_in, model_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(learning_rate=lr), metrics='accuracy')
    
    return model


build_fn().summary()

cv_preds = cross_validate(build_fn, verbose=True, random_state=2021, batch_size=16, 
                        stratify=True, num_folds=10, predict_train=True, predict_test=True, 
                        aug=True, aug_noise_std=0.03, num_augs=1,
                         save=True, save_filename_prefix='Try9 - FinalCV_1')