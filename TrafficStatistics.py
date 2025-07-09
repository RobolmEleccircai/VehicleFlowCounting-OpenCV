import cv2
import numpy as np

# 初始化全局变量
Video = cv2.VideoCapture('./CVideo.mp4')  # 接收读取的视频

# 创建mog对象
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# 返回指定形状和尺寸的结构元素--矩形
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 矩形轮廓最小尺寸
min_w = 100
min_h = 90

# 划线高度
line_high = 620

# 偏移量，线附近的一个范围
offset = 10

# 车辆数目
Number = 0

# 计算轮廓矩形的中心点
def CenterPointOperation(x, y, w, h):
    try:
        x1 = int(w / 2)
        y1 = int(h / 2)
        Center_X = int(x) + x1
        Center_Y = int(y) + y1
        return Center_X, Center_Y
    except Exception as e:
        print(f"Error in CenterPointOperation: {e}")
        return None, None

# 视频帧的图像图形学的运算处理
def Mathematical_Morphology(img):
    try:
        # 把原始帧进行灰度化，然后去噪
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        # 去噪——高斯滤波（常用的去噪方式）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imshow('blurred', blurred)
        # 前后景进行分离
        fgmask = fgbg.apply(blurred)
        cv2.imshow('fgmask', fgmask)
        # 腐蚀——去除高斯滤波后较为细小的噪声
        # 因为车辆形式识别出来的是 较为大块的图像，所以细小的噪声需要进行腐蚀化处理
        erode = cv2.erode(fgmask, kernel)
        cv2.imshow('erode', erode)
        # 膨胀，使目标增大,可添补目标中的空洞，把图像还原原尺寸
        dilate = cv2.dilate(erode, kernel, iterations=2)
        cv2.imshow('dilate', dilate)
        # 消除腐蚀膨胀处理后内部的小块
        # 闭运算——先膨胀后腐蚀，使轮廓变得光滑，弥合狭窄的间断，填充小的孔洞
        closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing', closing)
        return closing
    except Exception as e:
        print(f"Error in Mathematical_Morphology: {e}")
        return None

# 画出所有检测出来的轮廓并统计此视频帧中符合要求的车辆数目
def Draw_Contours(close, line_high, offset):
    try:
        cars = []
        CarNumber = 0
        # 查找轮廓
        Contours, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if Contours is None:
            print("No contours found.")
            return 0

        # 画出所有检测出来的轮廓
        for contour in Contours:
            # 最大外接矩形
            (x, y, w, h) = cv2.boundingRect(contour)
            # 过滤小框（小矩形），通过外接矩形的宽高大小来过滤掉小矩形
            is_valid = (w >= min_w) & (h >= min_h)
            if not is_valid:
                continue

            # 识别到的正常的车的轮廓
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

            # 把车抽象成一点，及轮廓矩形中心点
            # 通过轮廓矩形计算矩形的中心点
            CenterPoint = CenterPointOperation(x, y, w, h)
            if CenterPoint != (None, None):
                cars.append(CenterPoint)
                cv2.circle(frame, (CenterPoint), 5, (0, 0, 255), -1)

            # 判断汽车是否过检测线
            for(x, y) in cars:
                if y > (line_high - offset) and y < (line_high + offset):
                    # 落入了划线的范围区间内，计数+1
                    CarNumber += 1
                    cars.remove((x, y))

        return CarNumber
    except Exception as e:
        print(f"Error in Draw_Contours: {e}")
        return 0

# 主函数 循环读取视频帧
while True:
    try:
        # 读取到了视频帧，frame表示截取到一帧的图片
        ret, frame = Video.read()
        if not ret:
            print("Failed to read the frame or end of video.")
            break  # 读取失败或视频结束，退出

        # 画出检测线
        cv2.line(frame, (1280, line_high), (0, line_high), (255, 255, 0), 3)  # 使用黄色的线绘制检测线

        # 视频帧的图像图形学的运算处理
        close = Mathematical_Morphology(frame)
        if close is None:
            print("Error in processing frame.")
            continue  # 如果图像处理失败，跳过当前帧

        # 统计视频帧中符合要求的车辆数目
        Number = Number + Draw_Contours(close, line_high, offset)

        # 在帧上显示车辆数目
        cv2.putText(frame, 'Vehicle Number: ' + str(Number), (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # 显示帧
        cv2.imshow('Counting the number of vehicles', frame)

    except Exception as e:
        print(f"Error in main loop: {e}")
        break

    # 按ESC退出
    key = cv2.waitKey(5)  # 5ms一帧
    if key == 27:  # 按下ESC退出
        print("Exit on ESC key press.")
        break

# 释放资源
Video.release()
cv2.destroyAllWindows()
