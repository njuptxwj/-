import numpy as np
import cv2
from sklearn.cluster import KMeans
import turtle
import time
import pygame

# 检查颜色是否接近白色
def is_near_white(color, threshold=80):
    return all(c > threshold for c in color)

# 检查是否为接近黑色
def is_near_black(color, threshold=50):
    return all(c < threshold for c in color)

# 检查轮廓是否靠近图像边缘
def is_contour_near_border(contour, img_width, img_height, margin=10):
    for point in contour:
        x, y = point[0]
        if x < margin or y < margin or x > img_width - margin or y > img_height - margin:
            return True
    return False

# 计算区域跨度（最远点距离）
def calculate_span(points):
    if len(points) < 2:
        return 0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return (np.max(x_coords) - np.min(x_coords)) + (np.max(y_coords) - np.min(y_coords))

# 开始时间
start = time.time()
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("bgm.mp3")
pygame.mixer.music.play(-1, 0, 0)

# 读取并调整图片
img = cv2.imread('my.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
width, height = 400, 300
scale_factor = min(width / img.shape[1], height / img.shape[0])
scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# 将图片数据转换为一维数组
pixels = scaled_img.reshape((-1, 3))

# 使用 KMeans 聚类
n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_

# 初始化 turtle
turtle.setup(width=800, height=554)
turtle.bgcolor('white')
turtle.bgpic("bgpic.gif")
turtle.title('按图画图')
turtle.speed(0)

# 记录每个区域的颜色和面积
color_area = []
colors = []
for i in range(n_clusters):
    color_mask = (labels == i)
    mean_color = pixels[color_mask].mean(axis=0).astype(int)
    area = np.sum(color_mask)
    colors.append(mean_color)
    color_area.append((area, mean_color))

# 提取每个色块的轮廓信息
color_info = []
img_height, img_width = scaled_img.shape[:2]
for i, original_color in enumerate(colors):
    mask = (labels.reshape(scaled_img.shape[:2]) == i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        points = contour[:, 0]
        distance = calculate_span(points)
        # 黑色区域保留，即使面积小
        if is_near_black(original_color):
            color_info.append((len(points), original_color, distance, contour))
        # 对于非黑色区域，如果点数过少或者跨度大但点数分布稀疏，则忽略
        elif len(points) > 5 and distance > 8:
            color_info.append((len(points), original_color, distance, contour))

# 按距离排序，优先绘制大跨度区域
color_info.sort(key=lambda x: x[2], reverse=True)

# 开始绘制
print("开始绘制。。。")
border_margin = 5  # 边缘检测（像素）
x_offset = 130  # X轴偏移量
y_offset = 30   # Y轴偏移量
for area, original_color, distance, contour in color_info:
    # 如果是白色且靠近边缘，忽略绘制
    if is_near_white(original_color) and is_contour_near_border(contour, img_width, img_height, margin=border_margin):
        continue

    turtle_color = "#%02x%02x%02x" % (original_color[0], original_color[1], original_color[2])
    turtle.penup()
    x_start = contour[0][0][0] - img_width // 2+x_offset
    y_start = img_height // 2 - contour[0][0][1]+y_offset
    turtle.goto(x_start, y_start)
    turtle.pendown()
    turtle.color(turtle_color)
    turtle.begin_fill()

    for point in contour:
        x, y = point[0]
        x_turtle = x - img_width // 2+x_offset
        y_turtle = img_height // 2 - y+y_offset
        turtle.goto(x_turtle, y_turtle)

    turtle.end_fill()

# 隐藏 turtle
turtle.hideturtle()
print("绘制完成！！！")
pygame.mixer.music.stop()
time_used = time.time() - start
print("用时 %d 分 %d 秒" % (time_used // 60, time_used % 60))

# 保持窗口
turtle.done()
