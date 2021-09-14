import numpy as np
import cv2
import itertools

import c_rnn_model

class CRNN():
    def __init__(self):
        self.ctc_blank = 54    # c_rnn设置的参数，blank机制
        self.class_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                            'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
                            'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
                            'W': 30, 'X': 31, 'Y': 32, 'Z': 33, '云': 34, '京': 35, '川': 36, '晋': 37, '沪': 38, '津': 39,
                            '浙': 40, '粤': 41, '苏': 42, '豫': 43, '赣': 44, '鄂': 45, '闽': 46, '鲁': 47, '湘': 48, '渝': 49,
                            '贵': 50, '陕': 51, '皖': 52, '桂': 53}
        self.class_list = list(self.class_dictionary.keys())

        self.model = c_rnn_model.get_model(loss_model=False)
        self.model.load_weights('crnn_ep008-loss0.050-val_loss0.108.h5')


    # 调用c_rnn模型检测车牌数字字母
    # 输入：检测到的单张rgb车牌bounding box图像
    # 输出：该单张bounding box图像对应的字符
    def char_recognition(self, plate_image):

        plate_img1 = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        plate_img2 = cv2.resize(plate_img1, (256, 64), interpolation=cv2.INTER_AREA)

        pixel_max = np.max(plate_img2)
        pixel_min = np.min(plate_img2)
        plate_img3 = (plate_img2 - pixel_min) / (pixel_max - pixel_min)

        cv2.imshow('char', plate_img3)
        cv2.waitKey(1)

        plate_img4 = plate_img3.T
        plate_img5 = plate_img4[np.newaxis, :, :, np.newaxis]

        out1 = self.model.predict(plate_img5)    # (1, 32, 66)

        out2 = np.argmax(out1[0, 2:], axis=1)    # get max index -> len = 32
        out3 = [k for k, g in itertools.groupby(out2)]    # remove overlap value

        out4 = ''
        for j in range(len(out3)):

            index = int(out3[j])
            # blank机制去除冗余
            if index < self.ctc_blank:
                plate_char = self.class_list[index]
                out4 = out4 + str(plate_char)

        return out4








