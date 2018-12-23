

# #### 网络结构跟V9一致，各层添加了名字

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


# In[2]:

import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
# import cv2
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm


# In[3]:

from keras import optimizers


# In[4]:

im_width = 101
im_height = 101
im_chan = 1
basicpath = '/home/pengfei.liu/nlp/ziqi/TGS/data/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'


# In[5]:

img_size_ori = 101
img_size_target = 101

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res
    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]


# In[6]:

train_df = pd.read_csv("/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]


# In[7]:

train_df["images"] = [np.array(load_img("/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/train/images/{}.png".format(idx),color_mode = "grayscale")) / 255 for idx in tqdm(train_df.index)]


# In[8]:

train_df["masks"] = [np.array(load_img("/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/train/masks/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm(train_df.index)]


# In[9]:

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


# In[10]:

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)


# In[11]:

x_train.shape


# In[12]:

x_train2 = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train2 = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
print(x_train2.shape)
print(y_valid.shape)


# In[13]:

img_height=101
img_width=101
start_neurons=16
DropoutRatio=0.5


# In[14]:

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)


# In[15]:

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss


# In[16]:

def BatchActivate(x,name):
    x = BatchNormalization(name=name+'BN')(x)
    x = Activation('relu',name=name+'acti')(x)
    return x

def convolution_block(x, filters, size,name, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding,name=name+'cb_')(x)
    if activation == True:
        x = BatchActivate(x,name=name+'BA_')
    return x

def residual_block(blockInput,num_filters,name, batch_activate = False):
    x = BatchActivate(blockInput,name=name+'BA_')
    x = convolution_block(x, num_filters, (3,3),name=name+'cb_1_' )
    x = convolution_block(x, num_filters, (3,3), activation=False,name=name+'cb_2_')
    x = Add(name=name+'ad_')([x, blockInput])
    if batch_activate:
        x = BatchActivate(x,name=name+'BA_2_')
    return x


# In[17]:

img_height=101
img_width=101


# In[18]:

# Build model
def build_model(img_height,img_width, start_neurons,drop,DropoutRatio = 0.5):
    # 101 -> 50
    input_layer = Input((img_height,img_width, 1),name='input')
    x0_0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",name='x0_0_1')(input_layer)
    x0_0 = residual_block(x0_0,start_neurons * 1,name='x0_0_2')
    x0_0 = residual_block(x0_0,start_neurons * 1, batch_activate=True,name='x0_0_3')
    pool1 = MaxPooling2D((2, 2),name='x0_0_4')(x0_0)
    pool1 = Dropout(DropoutRatio/2,name='x0_0_5')(pool1) if drop else pool1

    # 50 -> 25
    x1_0 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",name='x1_0_1')(pool1)
    x1_0 = residual_block(x1_0,start_neurons * 2,name='x1_0_2')
    x1_0 = residual_block(x1_0,start_neurons * 2,batch_activate=True,name='x1_0_3')
    pool2 = MaxPooling2D((2, 2),name='x1_0_4')(x1_0)
    pool2 = Dropout(DropoutRatio,name='x1_0_5')(pool2) if drop else pool2

    # 25 -> 12
    x2_0 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",name='x2_0_1')(pool2)
    x2_0 = residual_block(x2_0,start_neurons * 4,name='x2_0_2')
    x2_0 = residual_block(x2_0,start_neurons * 4, batch_activate=True,name='x2_0_3')
    pool3 = MaxPooling2D((2, 2),name='x2_0_4')(x2_0)
    pool3 = Dropout(DropoutRatio,name='x2_0_5')(pool3) if drop else pool3

    # 12 -> 6
    x3_0 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",name='x3_0_1')(pool3)
    x3_0 = residual_block(x3_0,start_neurons * 8,name='x3_0_2')
    x3_0 = residual_block(x3_0,start_neurons * 8,batch_activate=True,name='x3_0_3')
    pool4 = MaxPooling2D((2, 2),name='x3_0_4')(x3_0)
    pool4 = Dropout(DropoutRatio,name='x3_0_5')(pool4) if drop else pool4

    # Middle
    x4_0 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same",name='x4_0_1')(pool4)
    x4_0 = residual_block(x4_0,start_neurons * 16,name='x4_0_2')
    x4_0 = residual_block(x4_0,start_neurons * 16,batch_activate=True,name='x4_0_3')
    
    # 6 -> 12
    x3_1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",name='x3_1_1')(x4_0)
    x3_1 = concatenate([x3_1, x3_0],name='x3_1_2')
    x3_1 = Dropout(DropoutRatio,name='x3_1_3')(x3_1) if drop else x3_1
    
    x3_1 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",name='x3_1_4')(x3_1)
    x3_1 = residual_block(x3_1,start_neurons * 8,name='x3_1_5')
    x3_1 = residual_block(x3_1,start_neurons * 8,batch_activate=True,name='x3_1_6')
    
    # 12 -> 25
    #x2_2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(x3_1)
    x2_2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid",name='x2_2_1')(x3_1)
    x2_2 = concatenate([x2_2, x2_0],name='x2_2_2')    
    x2_2 = Dropout(DropoutRatio,name='x2_2_3')(x2_2) if drop else x2_2
    
    x2_2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",name='x2_2_4')(x2_2)
    x2_2 = residual_block(x2_2,start_neurons * 4,name='x2_2_5')
    x2_2 = residual_block(x2_2,start_neurons * 4,batch_activate=True,name='x2_2_6')

    # 25 -> 50
    x1_3 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",name='x1_3_1')(x2_2)
    x1_3 = concatenate([x1_3, x1_0],name='x1_3_2')
        
    x1_3 = Dropout(DropoutRatio,name='x1_3_3')(x1_3) if drop else x1_3
    x1_3 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",name='x1_3_4')(x1_3)
    x1_3 = residual_block(x1_3,start_neurons * 2,name='x1_3_5')
    x1_3 = residual_block(x1_3,start_neurons * 2,batch_activate=True,name='x1_3_6')
    
    # 50 -> 101
    #x0_4 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(x1_3)
    x0_4 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid",name='x0_4_1')(x1_3)
    x0_4 = concatenate([x0_4, x0_0],name='x0_4_2')
    
    x0_4 = Dropout(DropoutRatio,name='x0_4_3')(x0_4) if drop else x0_4
    x0_4 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",name='x0_4_4')(x0_4)
    x0_4 = residual_block(x0_4,start_neurons * 1,name='x0_4_5')
    x0_4 = residual_block(x0_4,start_neurons * 1,batch_activate=True,name='x0_4_6')
    
    #x0_4 = Dropout(DropoutRatio/2)(x0_4)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(x0_4)
    output_layer = Conv2D(1, (1,1), padding="same", activation=None,name='out0')(x0_4)
    output_layer =  Activation('sigmoid',name='out1')(output_layer)
    model = Model(input_layer, output_layer)
    return model


# In[19]:

model1 = build_model(img_height,img_width, 16,True,0.5)
c = optimizers.adam(lr = 0.01)
model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
model1.summary()


# In[21]:

model_checkpoint = ModelCheckpoint('model_v1.h5',monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.1, patience=5, min_lr=0.00000001, verbose=1)

epochs = 1000
batch_size = 32
history = model1.fit(x_train2, y_train2,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                     shuffle=True,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                    verbose=1)


# In[22]:

model1 = load_model('model_v1.h5',custom_objects={'my_iou_metric': my_iou_metric})
# remove layter activation layer and use losvasz loss
input_x = model1.layers[0].input

output_layer = model1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])


# In[23]:

early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=30, verbose=1)
model_checkpoint = ModelCheckpoint('model_v1_1.h5',monitor='val_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True,save_weights_only=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.1, patience=4, min_lr=0.000000001, verbose=1)
epochs = 1000
batch_size = 32

history = model.fit(x_train2, y_train2,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle = True,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                    verbose=1)


# In[24]:

fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
ax_score.legend()


# In[27]:

model = load_model('model_v1_1.h5',custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})


# In[26]:

#model = load_model('model_v1.h5',custom_objects={'my_iou_metric': my_iou_metric})


# In[28]:

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2


# In[29]:

preds_valid = predict_result(model,x_valid,img_size_target)


# In[30]:

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in


    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
   # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


# In[31]:

thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm(thresholds)])
print(ious)


# In[32]:

threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()


# In[33]:

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[34]:

x_test = np.array([(np.array(load_img("/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


# In[35]:

x_test.shape


# In[36]:

test_df.index.shape


# In[37]:

preds_test = predict_result(model,x_test,img_size_target)


# In[38]:

import time
t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
t2 = time.time()
print(r"Usedtime = {t2-t1} s")


# In[40]:

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('./result/sub_v9_3_0911.csv')


# #### crf

# In[41]:

df = pd.read_csv('./result/sub_v9_3_0911.csv')


# In[42]:

df.head()


# In[43]:

def rle_encode(im):
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[44]:

def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


# In[45]:

test_path = '/home/pengfei.liu/nlp/ziqi/TGS/unet_resnet/test/images/'


# In[46]:

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)


# In[47]:

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral


# In[48]:

import numpy as np
from tqdm import tqdm
for i in tqdm(range(df.shape[0])):
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])        
        orig_img = imread(test_path+df.loc[i,'id']+'.png')        
        crf_output = crf(orig_img,decoded_mask)
        df.loc[i,'rle_mask'] = rle_encode(crf_output)


# In[49]:

df.to_csv('./result/sub_v9_3_crf_0911.csv',index=False)


# In[ ]:



