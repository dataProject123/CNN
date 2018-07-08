from PIL import Image
from PIL import ImageDraw
import os
import sys
from segment import Segment
from network import Network

now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录
print(now_path)
data_path = now_path + "data/"
model_path = now_path + "model/"

network_obj = Network(data_path = data_path, model_path = model_path)
network_obj.init()
#predict_value = network_obj.predictByImagePath(data_path + "training/0.png")
#print(predict_value)
#predict_value = network_obj.predictByImagePath(data_path + "training/1.png")
#print(predict_value)
#predict_value = network_obj.predictByImagePath(data_path + "training/152.png")
#print(predict_value)
#predict_value = network_obj.predictByImagePath(data_path + "training/148.png")
#print(predict_value)
#predict_value = network_obj.predictImagePath(data_path + "train/1.png")
#print(predict_value)

def get_operator(value):
    if value == 10:
        return '+'
    elif value == 11:
        return '-'
    elif value == 12:
        return '*'
    elif value == 13:
        return '/'
    else:
        return '未知'

def get_value(list):
    if list[1] == 10:
        return str(list[0] + list[1])
    elif list[1] == 11:
        return str(list[0] - list[1])
    elif list[1] == 12:
        return str(list[0] * list[1])
    elif list[1] == 13:
        return str(list[0] / list[1])
    else:
        return '未知'

file_path = now_path + "data/segment_test/test2.png"
segment_obj = Segment(file_path=file_path, ver_threshold = 0, hor_threshold = 0)
segment_obj.init()
sclice_list = segment_obj.get_sclice()
num = len(sclice_list)
list = [[] for i in range(num)]
for i in range(len(sclice_list)):
    img = sclice_list[i]
    predict_value = network_obj.predictByImageObj(img)
    list[i] = predict_value

print("%d %s %d = %s" % (list[0], get_operator(list[1]), list[2], get_value(list)))
