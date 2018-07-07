from PIL import Image
from PIL import ImageDraw
import os
import sys

now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录
create_path = now_path + "create/"
print(now_path)
test_image = Image.open(now_path + "data/segment_test/test.png")# 打开一张验证码
test_image = test_image.convert("L") # "L"表示灰度图
test_image.save(create_path + "gray.png") # 存储灰度图

def get_bin_table(thresold = 170):
    table = []
    for i in range(256):
        if i < thresold:
            table.append(0)
        else:
            table.append(1)
    return table # 得到的一个list，其0~thresold-1项为0，thresold~255项为1
table = get_bin_table() # thresold的值可以自行调节

bin_image = test_image.point(table, '1') # 用重新描点画图的方式得到二值图
bin_image.save(now_path + "create/binary.png") # 可省略，存储二值图

def vertical(img, threshold = 0):
    """传入二值化后的图片进行垂直投影"""
    pixdata = img.load()
    w,h = img.size
    ver_list = []
    # 开始投影
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        ver_list.append(black)
    #print(ver_list)
    #sys.exit()

    # 判断边界
    l,r = 0,0
    flag = False
    cuts = []
    for i,count in enumerate(ver_list):
        # 阈值这里为0
        if flag is False and count > threshold:
            l = i
            flag = True
        if flag and count < threshold:
            r = i-1
            flag = False
            cuts.append((l,r))
    return cuts

def horizontal(img, threshold = 0):
    """传入二值化后的图片进行横向投影"""
    pixdata = img.load()
    w,h = img.size
    hor_list = []
    # 开始投影
    for y in range(h):
        black = 0
        for x in range(w):
            if pixdata[x,y] == 0:
                black += 1
        hor_list.append(black)
    #print(hor_list)
    #sys.exit()

    # 判断边界
    t, b = 0,0
    flag = False
    cuts = []
    for i, count in enumerate(hor_list):
        # 阈值这里为0
        if flag is False and count > threshold:
            t = i
            flag = True
        if flag and count < threshold:
            b = i-1
            flag = False
            cuts.append((t, b))
    return cuts

def saveSclice(img, vertical, horizontal):
    w,h = img.size
    # 域由一个4元组定义，表示为坐标是 (left, top, right, bottom) 左上角为 (0, 0)的坐标系统
    for top, bottom in horizontal:
        for left, right in vertical:
            box = (left, top, right, bottom)
            child_image = img.crop(box) # 分割验证码图片
            child_image.resize((28, 28), Image.ANTIALIAS).save( create_path + str(left) + "_" + str(top) + "-" + str(right) + "-" + str(bottom) + ".png") # 存储分割后的图片 

ver_sclice = vertical(bin_image, 10)
print(ver_sclice)
hor_sclice = horizontal(bin_image, 10)
print(hor_sclice)
saveSclice(bin_image, ver_sclice, hor_sclice)
