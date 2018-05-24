from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, ZeroPadding2D
from keras.layers import Activation, BatchNormalization, add, Add, Concatenate, Multiply, Lambda
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.optimizers import Adam
from utils import multi_gpu_model
from utils import YoloLayer

import cv2
import numpy as np

import tensorflow as tf

import utils

class Detector:
    def __init__(self):
        self.batch_size = 4
        self.gpus = 1
        self.anchors = [21,38, 42,92, 58,191, 95,262, 117,119, 147,297, 221,350, 247,195, 353,357]
        self.labels = ["person","car","bus"]
        grid_scale = 1
        obj_scale = 5
        noobj_scale = 1
        xywh_scale = 1
        class_scale = 1
        ignore_thresh=0.5
        min_input_size = 384
        max_input_size = 448


        self.train_ints, self.valid_ints, self.labels, self.max_box_per_image = utils.create_training_instances(
            train_annot_folder='../data/voc2012_2/train/Annotations/',
            train_image_folder= '../data/voc2012_2/train/JPEGImages/',
            #train_annot_folder='/home/heecheol/Dataset/VOC2012/Annotations/',
            #train_image_folder= '/home/heecheol/Dataset/VOC2012/JPEGImages/',
            train_cache='VOC2012.pkl',
            valid_annot_folder='',
            valid_image_folder='',
            valid_cache='',
            labels=self.labels
        )
        if self.gpus>1:
            with tf.device('/cpu:0'):
                self.train_model, self.infer_model = self.HeeNet(
                    nb_class=len(self.labels),
                    anchors=self.anchors,
                    max_box_per_image=self.max_box_per_image,
                    max_grid=[max_input_size, max_input_size],
                    batch_size=self.batch_size // self.gpus,
                    ignore_thresh=ignore_thresh,
                    grid_scale=grid_scale,
                    obj_scale=obj_scale,
                    noobj_scale=noobj_scale,
                    xywh_scale=xywh_scale,
                    class_scale=class_scale,
                )
            self.train_model = multi_gpu_model(self.train_model,gpus=self.gpus)
        else:
            self.train_model, self.infer_model = self.HeeNet(
                    nb_class=len(self.labels),
                    anchors=self.anchors,
                    max_box_per_image=self.max_box_per_image,
                    max_grid=[max_input_size, max_input_size],
                    batch_size=self.batch_size,
                    ignore_thresh=ignore_thresh,
                    grid_scale=grid_scale,
                    obj_scale=obj_scale,
                    noobj_scale=noobj_scale,
                    xywh_scale=xywh_scale,
                    class_scale=class_scale,
                )


    def _inverted_res_block(self,inputs, expansion, stride, alpha, filters, block_id):
        in_channels = inputs._keras_shape[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None,
                       name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'expand_BN')(x)
            x = Activation(relu6, name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                            use_bias=False, padding='same',
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)

        x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and stride == 1:
            return Add(name=prefix + 'add')([inputs, x])

        return x

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _gru_block(self,x,Ht_1):
        shape=x[0].shape
        with tf.variable_scope('gru'):
            _conv2d_Z = Conv2D(320, (1, 1))
            _conv2d_R = Conv2D(320, (1, 1))
            _conv2d_H = Conv2D(320, (1, 1))
            output=[]
            for i in range(len(x)):
                concated_result = Concatenate(axis=-1)([Ht_1, x[i]])
                Zt = _conv2d_Z(concated_result)
                Zt = Activation('sigmoid')(Zt)
                Rt = _conv2d_R(concated_result)
                Rt = Activation('sigmoid')(Rt)
                Hht = Multiply()([Rt,Ht_1])
                Hht = Concatenate(axis=-1)([Hht,x[i]])
                Hht = _conv2d_H(Hht)
                Ht = Lambda(lambda x: 1 - x)(Zt)
                Ht = Multiply()([Ht_1,Ht])
                Hht = Multiply()([Zt,Hht])
                Ht = Add()([Ht,Hht])
                Ht_1 = Ht
                output.append(Ht)

            return output

    def _conv_block(self,inp, convs, do_skip=True):
        x = inp
        count = 0

        for conv in convs:
            if count == (len(convs) - 2) and do_skip:
                skip_connection = x
            count += 1

            if conv['stride'] > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(
                x)  # unlike tensorflow darknet prefer left and top paddings
            x = Conv2D(conv['filter'],
                       conv['kernel'],
                       strides = conv['stride'],
                       padding='valid' if conv['stride'] > 1 else 'same',
                       # unlike tensorflow darknet prefer left and top paddings
                       name='conv_' + str(conv['layer_idx']),
                       use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

        return add([skip_connection, x]) if do_skip else x

    def HeeNet(self,
               anchors,
               nb_class,
               max_box_per_image,
               max_grid,
               batch_size,
               ignore_thresh,
               grid_scale,
               obj_scale,
               noobj_scale,
               xywh_scale,
               class_scale,
               alpha=1.0,
               ):
        img_input = Input(shape=(None, None, 3))
        ground_truth= Input(shape=(None, None, len(anchors)//2, 4 + 1 + nb_class))# grid_h, grid_w, nb_anchor, 5+nb_class
        true_boxes=Input(shape=(1, 1, 1, max_box_per_image, 4))

        #Ht_1 = Input(shape=(input_shape[0]/32, input_shape[1]/32, 320))

        x=img_input

        first_block_filters = self._make_divisible(32 * alpha, 8)

        _conv2d = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2), padding='same',
                      use_bias=False, name='Conv1')
        _bnorm = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')
        _relu6 = Activation(relu6, name='Conv1_relu')

        x = _conv2d(x)
        x = _bnorm(x)
        x = _relu6(x)

        x = self._inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0)

        x = self._inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1)
        x = self._inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2)

        x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3)
        x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4)
        x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5)

        x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                                expansion=6, block_id=6)
        x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=7)
        x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=8)
        x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=9)

        x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=10)
        x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=11)
        x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=12)

        x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                                expansion=6, block_id=13)
        x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                expansion=6, block_id=14)
        x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                expansion=6, block_id=15)

        x = self._inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                                expansion=6, block_id=16)


        x = Conv2D(1280,
               kernel_size=1,
               use_bias=False,
               name='Conv_1')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = Activation(relu6, name='out_relu')(x)

        pred_layer = self._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                             {'filter': (9*(5+len(self.labels))), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], do_skip=False)



        loss_layer=(YoloLayer(anchors,
                            [1 * num for num in max_grid],
                            batch_size,
                            0,
                            ignore_thresh,
                            grid_scale,
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([img_input, pred_layer, ground_truth, true_boxes]))

        train_model = Model(inputs=[img_input,
                                    true_boxes,
                                    ground_truth],
                            outputs=[loss_layer])
        infer_model = Model(inputs=img_input, outputs=pred_layer)

        return [train_model, infer_model]

    def dummy_loss(self,y_true, y_pred):
        return tf.sqrt(tf.reduce_sum(y_pred))

    def DoTrain(self):
        #self.train_model.load_weights("../data/voc2012_2/train/notop.h5", by_name=True)
        self.train_model.load_weights("epoch_9.h5", by_name=True)

        learning_rate=1e-4
        optimizer = Adam(lr=learning_rate, clipnorm=0.001)
        self.train_model.compile(loss=self.dummy_loss, optimizer=optimizer)


        # delete difficult
        for inst_idx in range(len(self.train_ints)-1 , -1 , -1):
            inst = self.train_ints[inst_idx]
            for obj_idx in range(len(inst['object'])-1,-1,-1):
                obj = inst['object'][obj_idx]
                if obj['difficult'] == '1':
                    del self.train_ints[inst_idx]['object'][obj_idx]
            if inst['object']== []:
                del self.train_ints[inst_idx]
        for inst_idx in range(len(self.valid_ints)-1 , -1 , -1):
            inst = self.valid_ints[inst_idx]
            for obj_idx in range(len(inst['object'])-1,-1,-1):
                obj = inst['object'][obj_idx]
                if obj['difficult'] == '1':
                    del self.valid_ints[inst_idx]['object'][obj_idx]
            if inst['object']== []:
                del self.valid_ints[inst_idx]

        train_generator = utils.BatchGenerator(
            instances=self.train_ints,
            anchors=self.anchors,
            labels=self.labels,
            downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
            max_box_per_image=self.max_box_per_image,
            batch_size=int(self.batch_size),
            min_net_size=416-64,
            max_net_size=416+64,
            shuffle=True,
            jitter=0.3,
            norm=self.normalize)
        valid_generator = utils.BatchGenerator(
            instances= self.valid_ints,
            anchors=self.anchors,
            labels=self.labels,
            downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
            max_box_per_image=self.max_box_per_image,
            batch_size=int(self.batch_size),
            min_net_size=416,
            max_net_size=416,
            shuffle=True,
            jitter=0.0,
            norm=self.normalize)

        for i in range(10):
            self.train_model.fit_generator(
                generator=train_generator,
                steps_per_epoch=len(train_generator),
                epochs= 10,
                verbose=2,
                validation_data=valid_generator
            )
            self.train_model.save('epoch_'+str(i)+'.h5')


    def DoTest(self):
        self.train_model.load_weights("epoch_9.h5", by_name=True)

        obj_thresh, nms_thresh = 0.5, 0.45
        print(self.train_ints[0])

        import time
        for i in range(20):
            img_name ='/home/heecheol/Dataset/KNU_Campus/20180312_171706/20180312_171706_'
            num = str(i)
            while len(num)<4:
                num= '0'+num


            #image = cv2.imread(self.valid_ints[i]['filename'])

            image = cv2.imread(img_name+num+'.jpg')
            image_h, image_w, _ = image.shape

            new_image = utils.preprocess_input(image, 416, 416)
            new_image = np.expand_dims(new_image, 0)

            start_time= time.time()
            pred = self.infer_model.predict(new_image)
            print(time.time()-start_time)
            boxes = []

            boxes += utils.decode_netout(pred[0], self.anchors, obj_thresh, nms_thresh, 832, 832)
            # correct the sizes of the bounding boxes
            utils.correct_yolo_boxes(boxes, image_h, image_w, 832, 832)

            # suppress non-maximal boxes
            utils.do_nms(boxes, nms_thresh)

            # draw bounding boxes on the image using labels
            utils.draw_boxes(image, boxes, self.labels, obj_thresh)
            cv2.imwrite('vali_img_'+str(i)+'.jpg',image)


    def normalize(self, image):
        return (image / 128.) - 1.
