# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 00:03:46 2022

@author: D_Amr_Atif
"""

from numpy import mean, flatnonzero
from sklearn.model_selection import cross_val_score
import time
import numpy as np
from keras.utils.vis_utils import plot_model
import os
from keras.callbacks import EarlyStopping
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D,MaxPooling1D
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
# from keras import backend as K
from keras import metrics as tf_met
# from keras import optimizers as opt
# from keras import losses as los
# from tensorflow_addons import metrics as tfa_met
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,roc_curve

class CNN: 
        
        # Length of columns (features)
        # MAX_SEQUENCE_LENGTH = 1000
        # Dimensions
        EMBEDDING_DIM = 100       
        def __init__(self, x_train, x_test, x_val, y_train, y_test, y_val, word_ind):
                self.x_train = x_train
                self.x_test = x_test
                self.x_val = x_val
                self.y_train = y_train
                self.y_test = y_test
                self.y_val = y_val
                self.word_ind = word_ind
                
                self.fold = 5
                self.grid_search = False
                self.cross_validation = False
                self.counter = 0
        # def recall_m(self,y_true, y_pred):
        #         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        #         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        #         recall = true_positives / (possible_positives + K.epsilon())
        #         return recall

        # def precision_m(self,y_true, y_pred):
        #         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        #         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        #         precision = true_positives / (predicted_positives + K.epsilon())
        #         return precision

        # def f1_m(self,y_true, y_pred):
        #         precision = self.precision_m(y_true, y_pred)
        #         recall = self.recall_m(y_true, y_pred)
        #         return 2*((precision*recall)/(precision+recall+K.epsilon()))   
            
            
        def _fit__(self, solution=None, minmax=0):
              
                # cols = flatnonzero(solution)

                # k = 5
                # if self.grid_search:
                #         knn_gs = KNeighborsClassifier()                                 
                #         param_grid = {"n_neighbors": arange(1, 10)}             #create a dictionary of all values we want to test for n_neighbors
                #         knn_gscv = GridSearchCV(knn_gs, param_grid, cv=self.kfold)   #use gridsearch to test all values for n_neighbors
                #         knn_gscv.fit(feat, label)                               #fit model to data
                #         k = knn_gscv.best_params_["n_neighbors"]                #check top performing n_neighbors value
                
                # clf = KNeighborsClassifier(n_neighbors=k)

                # train_data = self.X_train[:,cols]
                # test_data = self.X_test[:,cols]
                
                # clf.fit(train_data, self.y_train)
                
                # if self.cross_validation:
                #         cv_scores = cross_val_score(clf, self.feat[:,cols], label, cv=self.kfold)
                #         err = [1 - x for x in cv_scores]
                # else:
                #         score = clf.score(test_data, self.y_test)
                #         err = 1 - score
                
                # return mean(err)  

                #############################################################
                #############################################################
                
                cols = flatnonzero(solution)
                # print(cols.size)
                # %%
                #Using Pre-trained word embeddings
                GLOVE_DIR = "datasets/UCI/Fake_True_News/glove.6B" 
                embeddings_index = {}
                f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")    
                
                #############################################################
                # for each line in the file
                for line in f:    
                    # Split the line into a list of words by space
                    values = line.split()
                    #print(values[1:])
                    
                    # the first word
                    F_word = values[0]
                    
                    # coefs: the remaining values in the line
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[F_word] = coefs
                #############################################################    
                f.close()
                    
                # print('Total %s word vectors in Glove.' % len(embeddings_index))    
                #############################################################
                # matrix with rows =  len(word_index) + 1 , columns = EMBEDDING_DIM = 100
                embedding_matrix = np.random.random((len(self.word_ind) + 1, self.EMBEDDING_DIM))
                
                #############################################################
                for word, i in self.word_ind.items():
                        
                    # search the coefs of word in embeddings_index 
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = embedding_vector
                #############################################################
                
                embedding_layer = Embedding(len(self.word_ind) + 1,
                                            self.EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=cols.size)
                            
                #############################################################
                #############################################################
                # %%
                # Simple CNN model
                sequence_input = Input(shape=(cols.size,), dtype='int32')
                embedded_sequences = embedding_layer(sequence_input)
                l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
                l_pool1 = MaxPooling1D(5, padding='same')(l_cov1)
                l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
                l_pool2 = MaxPooling1D(5, padding='same')(l_cov2)
                l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
                l_pool3 = MaxPooling1D(35, padding='same')(l_cov3)  # global max pooling
                l_flat = Flatten()(l_pool3)
                l_dense = Dense(128, activation='relu')(l_flat)
                preds = Dense(2, activation='softmax')(l_dense)
                   
                clf = Model(sequence_input, preds)
                
                # clf.compile(loss='categorical_crossentropy',
                #             optimizer='adam',
                #             metrics=['acc',self.f1_m, self.precision_m, self.recall_m])

                
                clf.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['acc',
                                     tf_met.Precision(),
                                     tf_met.Recall()])
                
                  # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  #    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  #    metrics=[tf.keras.metrics.Accuracy(),
                  #             tf.keras.metrics.Precision(),
                  #             tf.keras.metrics.Recall(),
                  #             tfa.metrics.F1Score(num_classes=nb_classes,
                  #                                 average='macro',
                  #                                 threshold=0.5))
                  
                  
                # print("Fitting the simple convolutional neural network model classifier")
                # clf.summary()
                self.counter += 1
                model_path = os.getcwd() + "/Final_AVO/model_" + str(self.counter) + ".png"    
                plot_model(clf, to_file=model_path, show_shapes=True)
                                     
                
                train_data = self.x_train[:,cols]
                test_data = self.x_test[:,cols]
                val_data = self.x_val[:,cols]
                # print("train_data:", train_data.shape) 
                # print("test_data:", test_data.shape)
                # print("val_data:", val_data.shape)
                
                # start = time.time()
                #############################################################
                # history = clf.fit(train_data, self.y_train, validation_data=(val_data, self.y_val),
                #                   epochs=30, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
                history = clf.fit(train_data, self.y_train, validation_data=(val_data, self.y_val),
                                  epochs=30, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
                
                

                #############################################################
                # end = time.time()
                           
                #############################################################
                #############################################################
                # %%                    
                # %matplotlib inline 
                # list all data in history
                print(history.history.keys())
                # summarize history for accuracy
                plt.semilogy(history.history['acc'])
                plt.semilogy(history.history['val_acc'])
                plt.title('Model Accuracy')
                plt.axis([0, 14, 0, 1])   # for set values of x-axis forst from "0 to 50" and then y-axis from "0 to 1" as we wants
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='best')
                plt.show()
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.axis([0, 14, 0, 3])    # for set values of x-axis forst from "0 to 50" and then y-axis from "0 to 1" as we wants
                plt.plot(history.history['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='best')
                plt.show()
                    
                
                
                # %%
                # print('Training Time: ', end - start)
                # score 
                loss, accuracy, Precision_res, Recall_res = clf.evaluate(test_data, self.y_test)

                err = 1 - accuracy
                F1_score_res = 2*((Precision_res*Recall_res)/(Precision_res+Recall_res)) 
                
                # print(loss, accuracy, F1_score_res, Precision_res, Recall_res)
                if self.cross_validation:
                    cv_scores = cross_val_score(clf, self.feat[:,cols], label, cv=self.kfold)
                    err = [1 - x for x in cv_scores]
                else:
                   
                    score = clf.score(test_data, self.y_test)
                    err = 1 - score

                        
                return mean(err), mean(F1_score_res), mean(Precision_res), mean(Recall_res)


                # if self.cross_validation:
                #         cv_scores = cross_val_score(clf, train_data, self.y_train, cv=self.kfold)
 
                #         err = [1 - x for x in cv_scores]
                # else:
                #         score = clf.score(test_data, self.y_test)
                #         # accuracy = accuracy_score(self.y_test, y_pred)

                #         err = 1 - score
                
                # return mean(err)               
                
                ##################################################################
                # print("Accuracy: %.2f%%" % (accuracy * 100.0))
                # confusion_matrix(self.y_test,y_pred)
                
                # drawing confusion matrix
                # pd.crosstab(self.y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True)
                ##################################################################
                # y_pred = clf.predict(test_data)
                
                
                # F1_score_res = f1_score(self.y_test, y_pred, average='binary')
                # Precision_res = precision_score(self.y_test, y_pred, average='binary')
                # Recall_res = recall_score(self.y_test, y_pred, average='binary')
                # kappa_res = cohen_kappa_score(self.y_test, y_pred)
                
                
                # cm1 = confusion_matrix(self.y_test,y_pred)
                # Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                # Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                
                
                
                # # cm5 = pd.DataFrame(confusion_matrix(self.y_test,y_pred), index = labels, columns = labels)

                # # plt.figure(figsize = (10, 8))
                # # sns.heatmap(cm5, annot = True, cbar = False, fmt = 'g')
                # # plt.ylabel('Actual values')
                # # plt.xlabel('Predicted values')
                # # plt.show()

                # # fpr_res, tpr_res, _ = roc_curve(self.y_test, y_pred)
                # ##################################################################
                # # print(Specificity, Sensitivity)
                # return err, kappa_res, Precision_res, Recall_res, F1_score_res, Specificity, Sensitivity
