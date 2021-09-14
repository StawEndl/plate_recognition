import numpy as np
import cv2
from crnn_recognition import CRNN
from yolo import YOLO
import tensorflow as tf
import PIL.Image as Image
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

    # 调用海康摄像头配置
    cap = cv2.VideoCapture("rtsp://admin:123456abc@192.168.1.120/h264/ch1/main/av_stream")
    # cap = cv2.VideoCapture(0)

    frame_height = cap.get(3)    # 3代表帧的宽度， 2560.0
    frame_width = cap.get(4)     # 4代表帧的高度， 1440.0
    frame_fps = cap.get(5)       # 5代表帧速FPS， 8.0
    print(frame_fps, frame_width, frame_height)    # 8.0 1440.0 2560.0

    # mtx  : 相机内参矩阵
    # dist : 相机畸变系数
    mtx = np.array([[8.45559259e+02, 0.00000000e+00, 1.32460030e+03],
                    [0.00000000e+00, 8.50592806e+02, 7.29938055e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[-1.12961572e-01, 1.54572487e-02, -1.66183578e-03, -1.09252844e-04, -1.15994476e-03]])
    w = 2560
    h = 1440
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.1, (w, h))
    crnn = CRNN()
    yolo = YOLO()
    step = 0
    while True:
        plate_number = []
        ret, frame = cap.read()    # (1440, 2560, 3)
        print(ret)
        if not ret:
            continue
        step = step + 1
        if step != 5:
            continue
        else:
            step = 0

        frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        r_image, j, part_imgs = yolo.detect_image(frame)

        for part_img in part_imgs:
            res = crnn.char_recognition(part_img)
            print("res", res)
            plate_number.append(res)

        print("plates number is ", len(plate_number))
        for ind_plate in range(len(plate_number)):
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)
            print(plate_number)


        frame = np.array(r_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("raw_img", cv2.resize(frame, (256*3, 144*3)))
        # cv2.imshow('un_distort_img', cv2.resize(dst, (256*3, 144*3)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()






