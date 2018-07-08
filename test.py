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
predict_value = network_obj.predict(data_path + "train/1.png")
print(predict_value)
predict_value = network_obj.predict(data_path + "train/1.png")
print(predict_value)
predict_value = network_obj.predict(data_path + "train/1.png")
print(predict_value)
predict_value = network_obj.predict(data_path + "train/1.png")
print(predict_value)
predict_value = network_obj.predict(data_path + "train/1.png")
print(predict_value)

#file_path = now_path + "data/segment_test/test.png"
#segment_obj = Segment(file_path=file_path)
#segment_obj.init()
#sclice_list = segment_obj.get_sclice()
#print(len(sclice_list))
