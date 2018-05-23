import keras

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape, Add
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras import backend as K
import Detector



if __name__ == '__main__':
    # model2 = keras.applications.MobileNetV2(include_top=False)
    # yaml_string = model2.to_yaml()  # 모델 아키텍처를 yaml 형식으로 저장
    # with open("model2.yaml", "w") as json_file:
    #     json_file.write(yaml_string)
    # model2.save('notop.h5')
    detector=Detector.Detector()
    detector.DoTrain()
    #detector.DoTest()
    # model2=MobileNetv2((None, None, 3))
    # model2.load_weights('Weights/mobilenet_v2_notop.h5')
    # print(model2.output)

