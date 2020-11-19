import os
# Filter out INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tkinter import *
import tensorflow as tf
import cv2
import numpy as np
from tkinter import messagebox as msg
import datetime
import playsound

root = Tk()
root.title("상황판")
root.geometry("640x400+1032+512")
root.configure(bg="yellow")
root.resizable(False, False)

Label(root, text="비밀번호 인식", font=("times new roman", 30, "bold"), bg="black", fg="red").pack()

lblCount = Label(root, text="인식된 숫자 →", font=("times new roman", 20, "bold"), bg="yellow", fg="red")
lblCount.place(x=30, y=120)

lblCorrected = Label(root, text="입력된 비밀번호 →", font=("times new roman", 20, "bold"), bg="yellow", fg="red")
lblCorrected.place(x=30, y=60)

def draw_rectangle_and_text(img_color):
    x, y = 25, 25
    cv2.rectangle(img_color, (450, 250), (width - 50, height - 50), (0, 0, 255), 3)
    cv2.rectangle(img_color, (450 + x, 250 + y), (width - 50 - x, height - 50 - y), (0, 255, 0), 2)

    b, g, r = 0, 102, 204
    cv2.putText(img_color, "Use the space bar", (390, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (b, g, r), 1,
                cv2.LINE_AA)
    cv2.putText(img_color, "Above the green rectangle borderline", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (b, g, r), 1,
                cv2.LINE_AA)

def initial():
    if len(corrected) > 0:
        corrected.clear()
        lblCorrected['text'] = '입력된 비밀번호 →'
        msg.showinfo('info', '초기화가 되었습니다.')

def process(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소에 있어 선호되는 방법
    # gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 가운데 픽셀에 가장 큰 가중치를 두어서 계산을 합니다
    # (val, val2)와 같이 두 개의 값이 달라도 되지만, 모두 양의 홀수이어야함
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)  # default
    # THRESH_OTSU + threshold -> 이진화합니다. 이미지에 대한 히스토그램에서 임계값을 자동으로 계산해줍니다. -> 이때 임계값은 0으로 전달하면 됩니다.
    (thresh, img_binary) = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w = img_binary.shape
    print("h:", h, "w:", w)
    ratio = 100 / h
    new_h = 100
    new_w = w * ratio
    print(ratio, new_w, new_h)

    img_empty = np.zeros((110, 110), dtype=img_binary.dtype)

    # 확대일 경우 cv2.INTER_NEAREST <- 비슷한 결과 반환
    # 이웃 보간법의 장점은 구현하기가 쉽고 동작이 빠르다는 점이다.
    # 그러나 이 방법을 이용하여 확대된 결과 영상은 픽셀 값이 급격하게 변화하는 것 같은 계단 현상(또는 블록 현상)이 나타날 수 있다는 단점이 있다.
    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], :img_binary.shape[1]] = img_binary
    img_binary = img_empty

    # 컨투어(contour)란 동일한 색 또는 동일한 픽셀값(강도,intensity)을 가지고 있는 영역의 경계선 정보다.
    # 물체의 윤곽선, 외형을 파악하는데 사용된다.
    # OpenCV의 findContours 함수로 이미지의 컨투어 정보, 컨투어의 상하구조(hierachy) 정보를 출력한다.
    # 흑백이미지 또는 이진화된 이미지만 적용할 수 있다.

    # cv2.findContours(image, mode, method)
    # image: 흑백이미지 또는 이진화된 이미지
    # mode : 컨투어를 찾는 방법
    # cv2.RETR_EXTERNAL: 컨투어 라인 중 가장 바깥쪽의 라인만 찾음
    # cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음
    # cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함
    # cv2.RETR_TREE: 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함
    # method : 컨투어를 찾을 때 사용하는 근사화 방법
    # cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
    # cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환
    # cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용하여 컨투어 포인트를 줄임
    # cv2.CHAIN_APPROX_TC89_KCOS: Teh_Chin 연결 근사 알고리즘 KCOS 버전을 적용하여 컨투어 포인트를 줄임

    # .copy() -> 원본 이미지를 변경시키기 때문에
    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 이미지 모멘트(Image Moments)
    # 이미지 모멘트는 객체의 무게중심, 객체의 면적 등과 같은 특성을 계산할 때 유용합니다.
    # OpenCV의 cv2.moments() 함수는 이미지 모멘트를 계산하고 이를 사전형 자료에 담아 리턴합니다.
    # 모멘트 종류는 3가지이며 아래와 같이 총 24개의 값을 가집니다.
    # 공간 모멘트(Spatial Moments)
    # m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
    # 중심 모멘트(Central Moments)
    # mu20, mu11, mu02, mu30, mu21, mu12, mu03
    # 평준화된 중심 모멘트(Central Normalized Moments)
    # nu20, nu11, nu02, nu30, nu21, nu03

    # x, y, w, h = cv2.boundingRect(cnts[0][0])
    # x0, y0 = zip(*np.squeeze(cnts[0][0]))
    # plt.plot(x0, y0, c="b")
    # plt.plot(
    #     [x, x + w, x + w, x, x],
    #     [y, y, y + h, y + h, y],
    #     c="r"
    # )
    # plt.show()

    # 컨투어의 무게중심 좌표를 구합니다.
    M = cv2.moments(cnts[0][0])
    center_x = (M["m10"] / M["m00"])
    center_y = (M["m01"] / M["m00"])

    # 무게 중심이 이미지 중심으로 오도록 이동시킵니다.
    height, width = img_binary.shape[:2]
    shiftx = width / 2 - center_x
    shifty = height / 2 - center_y
    print("cx:", center_x, "cy:", center_y)
    print("shiftx:", shiftx, "shifty:", shifty)

    move = np.float32([[1, 0, shiftx], [0, 1, shifty]])

    # [ [ 1, 0, x축의 이동 길이], <- 2x3
    # [0, 1, y축의 이동 길이] ]

    img_binary = cv2.warpAffine(img_binary, move, (width, height))

    # 비교를 위해 이미지 축소
    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow('Comparison', img_binary)

    # 이미지를 0 ~ 1 사이 값을 갖는 크기 784(=28x28)의 일차원 배열로 변환
    flatten = img_binary.flatten() / 255.0

    return flatten

def speak(case):
    path = 'samples/'
    if case == 0:
        playsound.playsound(path + "0.mp3")
    elif case == 1:
        playsound.playsound(path + "1.mp3")
    elif case == 2:
        playsound.playsound(path + "2.mp3")
    elif case == 3:
        playsound.playsound(path + "3.mp3")
    elif case == 4:
        playsound.playsound(path + "4.mp3")
    elif case == 5:
        playsound.playsound(path + "5.mp3")
    elif case == 6:
        playsound.playsound(path + "6.mp3")
    elif case == 7:
        playsound.playsound(path + "7.mp3")
    elif case == 8:
        playsound.playsound(path + "8.mp3")
    elif case == 9:
        playsound.playsound(path + "9.mp3")
    elif case == 90:
        playsound.playsound(path + "90.mp3") # 인식
    elif case == 91:
        playsound.playsound(path + "91.mp3") # 확인
    elif case == 92:
        playsound.playsound(path + "92.mp3") # 초기화
    elif case == 93:
        playsound.playsound(path + "93.mp3") # 캡쳐

# DirectShow (via videoInput)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4

lotto = [0, 4, 2, 3]
corrected = []
failure = 0

manual = '''설명서
Space bar: 인식
Backspace: 초기화
Enter: 확인   Tab: 캡쳐
Esc: 종료
실패 횟수 3번 -> 자동 종료
'''

lblManual = Label(root, text=manual, font=("times new roman", 20, "bold"), justify=LEFT, bg="yellow", fg="#ff3399")
lblManual.place(x=30, y=200)

# C:\Users\iotpc\PycharmProjects\anaconda\samples
path = 'samples/p_v1.png'
photo = PhotoImage(file=path)
Label(root, image=photo).pack(side='right')

lblText = Label(root, text="실패 횟수 →", font=("times new roman", 20, "bold"), bg="yellow", fg="red")
lblText.place(x=30, y=160)
lblFailing = Label(root, text=(failure, '/ 3'), font=("times new roman", 20, "bold"), bg="yellow", fg="red")
lblFailing.place(x=200, y=160)

while True:
    ret, img_color = cap.read()

    if ret == False:
        break

    img_input = img_color.copy()
    draw_rectangle_and_text(img_color)
    cv2.imshow('Webcam', img_color)

    img_roi = img_input[250:height - 50, 450:width - 50]
    key = cv2.waitKey(1)

    if key == 8:
        if len(corrected) > 0:
            speak(92)
            initial()
    elif key == 9:
        if len(corrected) > 0:
            speak(93)
            now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            path = 'samples/' + str(now) + ".png"
            cv2.imwrite(path, img_roi)
    elif key == 13:
        if len(corrected) > 0:
            speak(91)
            for x in corrected:
                speak(x)
            check = lotto == corrected
            if check == True:
                msg.showinfo('info', '일치')
                break
            else:
                failure += 1
                lblFailing.config(text=(failure, '/ 3'))
                msg.showerror('error', '불일치')
            if failure == 3:
                break
    elif key == 27:
        break
    elif key == 32:
        '''
        img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cv2.rectangle(img_roi, (x, y,), (x+w, y+h), (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(img_roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(x, y, w, h, x+w, y+h)

        img_bis_input = img_roi.coy()
        img_bis_roi = img_roi[y: y + h, x: w + x]

        flatten = process(img_bis_roi)
        '''
        matrix = 1024

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(matrix, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.load_weights('model_11_03_15_23')

        flatten = process(img_roi)

        # 인풋 샘플에 대한 아웃풋 예측을 생성합니다.
        # 데이터에 지정된 수의 차원을 맞추기 위해

        predictions = model.predict(flatten[np.newaxis, :])
        # print("flatten: ", flatten) -> flatten.shape (784,)
        # print("flatten[np.newaxis, :].shape", flatten[np.newaxis, :].shape) -> flatten[np.newaxis, :].shape (1, 784)

        # Tensorflow 2.0에서 Tensorflow 1.x 코드 실행하기
        # 2차원 배열의 경우  최대 인덱스 개수는 2(<= rank) -> 두번째 인자로 0과 1을 사용할 수 있습니다.
        # 0 -> 열에서 가장 큰 값 찾기, 1 -> 행에서 가장 큰 값 찾기

        with tf.compat.v1.Session() as sess:
            print("예측:", tf.argmax(predictions, 1).eval())
            lblCount['text'] = '인식된 숫자 →', tf.argmax(predictions, 1).eval()

            temp = str(tf.argmax(predictions, 1).eval())
            case = 90
            speak(case)
            case = int(temp[1:2])
            speak(case)

            if len(corrected) < 4:
                corrected.append(int(temp[1:2]))
                lblCorrected.config(text=('입력된 비밀번호 →', corrected))
            else:
                msg.showwarning('warning', '비밀번호는 4자리입니다.')

        root.update()
        cv2.imshow('Chopthescreen', img_roi)

        # 키보드의 키 입력을 무한히 기다리며 cap.read() <- 웹캠을 사용하여 프레임을 새로 고치지 않으므로 화면이 일시 중지됩니다.
        # cv2.waitKey(0)

root.destroy()
cap.release()
cv2.destroyAllWindows()

