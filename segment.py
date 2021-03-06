from PIL import Image
from PIL import ImageDraw
import os
import sys

class Segment:
    def __init__(self, file_path, bin_threshold = 170, ver_threshold = 10, hor_threshold = 10):
        self.bin_threshold = bin_threshold
        self.ver_threshold = ver_threshold
        self.hor_threshold = hor_threshold
        self.file_path = file_path
    
    def init(self):
        self.img = Image.open(self.file_path)
        self.gray_img = self.img.convert("L") # "L"表示灰度图
        self.table = self.get_bin_table() # thresold的值可以自行调节
        self.table2 = self.get_bin_table2() # thresold的值可以自行调节
        #self.bin_img  = self.gray_img.point(self.table2, '1') # 用重新描点画图的方式得到二值图
        self.bin_img = self.gray_img
        self.width, self.high = self.bin_img.size
        self.pixdata = self.bin_img.load()
        self.horizontal = self.get_horizontal()
        self.vertical = self.get_vertical()
       
        return True
        
    # 得到的一个list，其0~threshold项为1，threshold~255项为0
    def get_bin_table(self):
        table = []
        for i in range(256):
            if i > self.bin_threshold:
                table.append(0)
            else:
                table.append(1)
        return table 
    
    # 得到的一个list，其0~threshold项为0，threshold~255项为1
    def get_bin_table2(self):
        table = []
        for i in range(256):
            if i < self.bin_threshold:
                table.append(0)
            else:
                table.append(1)
        return table 
    
    def get_vertical(self):
        """传入二值化后的图片进行垂直投影"""
        ver_list = []
        # 开始投影
        for x in range(self.width):
            special = 0
            for y in range(self.high):
                if self.pixdata[x,y] > 0:
                    special += 1
            ver_list.append(special)
        
        #print(ver_list)
        # 判断边界
        l,r = 0,0
        flag = False
        cuts = []
        for i,count in enumerate(ver_list):
            if flag is False and count > self.ver_threshold:
                l = i
                flag = True
            if flag and count <= self.ver_threshold:
                r = i-1
                flag = False
                cuts.append((l,r))
        
        return cuts
    
    def get_horizontal(self):
        """传入二值化后的图片进行横向投影"""
        hor_list = []
        # 开始投影
        for y in range(self.high):
            special = 0
            for x in range(self.width):
                if self.pixdata[x,y] > 0:
                    special += 1
            hor_list.append(special)
    
        # 判断边界
        t, b = 0,0
        flag = False
        cuts = []
        for i, count in enumerate(hor_list):
            if flag is False and count > self.hor_threshold:
                t = i
                flag = True
            if flag and count <= self.hor_threshold:
                b = i-1
                flag = False
                cuts.append((t, b))
        return cuts
    
    def get_sclice(self):
        # 域由一个4元组定义，表示为坐标是 (left, top, right, bottom) 左上角为 (0, 0)的坐标系统
        sclice = []
        for top, bottom in self.horizontal:
            for left, right in self.vertical:
                width_buffer = 2
                high_buffer = 2
                sclice_width = right - left + 1
                sclice_high = bottom - top + 1
                ratio_width = (28 - width_buffer * 2)/sclice_width
                ratio_high = (28 - high_buffer * 2)/sclice_high
                ratio_min = min(ratio_width, ratio_high)
                new_width = int(ratio_min * sclice_width)
                new_high = int(ratio_min * sclice_high)
                
                # 分割图片框
                box = (left, top, right, bottom)
                # 分割提取图片
                child_image = self.bin_img.crop(box)
                # 等比例缩放
                child_image = child_image.resize((new_width, new_high), Image.ANTIALIAS)
                # 新建一张底色图片
                new_img = Image.new("RGBA", (28, 28), (0, 0, 0))
                new_img = new_img.convert("L")
                #new_img  = new_img.point(self.table2, '1')
                new_left = int(14 - new_width/2)
                new_right = int(14 + new_width/2)
                new_top = int(14 - new_high/2)
                new_bottom = int(14 + new_high/2)
                # 将底图和切割图进行合并处理
                new_img.paste(child_image, (new_left, new_top, new_right, new_bottom))
                sclice.append(new_img)
        
        return sclice

if __name__ == '__main__':
    now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录
    print(now_path)
    #file_path = now_path + "data/segment_test/test.png"
    file_path = now_path + "data/segment_test/test3.png"
    segment_obj = Segment(file_path=file_path, ver_threshold = 0, hor_threshold = 0)
    segment_obj.init()
    sclice_list = segment_obj.get_sclice()
    print(len(sclice_list))
    for i in range(len(sclice_list)):
        img = sclice_list[i]
        img.save("create/" + str(i) + ".png")
