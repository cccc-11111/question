import csv
from collections import Counter

import torch

from ultralytics import YOLO
import os

torch.cuda.empty_cache()

if __name__ == '__main__':
    model_path = r'F:\experiment\原1\蝴蝶兰实验\训练1（batch=64）\weights\best.pt'
    test_img_path = r'F:\experiment\原1\蝴蝶兰实验\test2'

    '''结果保存文件路径创建'''
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    '''CSV文件创建'''
    csv_file = os.path.join(save_dir, 'result.csv')
    '''txt文件创建'''
    txt_file = os.path.join(save_dir, 'result.txt')


    cls = []  # 图像识别类别
    all_cls_dict = {}   #图像识别统计字典
    total_count = 0
    model = YOLO(model_path)  #模型加载
    results = model.predict(test_img_path, project='runs/segment/detect', save=True, conf=0.5, device='cuda:0')  #进行预测

    '''---------------------------进行识别类型计数----------------------------------------'''
    if results:
        #打开CSV文件准备写入
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['图片序号', '图片文件名', '识别类型', '检测数量']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for i in range(len(results)):
                result = results[i]
                #获取图片文件名
                img_filename = os.path.basename(result.path) if hasattr(result, 'path') else f'image_{i+1}'
                if result.boxes is not None:
                    cls.append([int(x) for x in result.boxes.cls.tolist()])  # 把每个图像中的识别结果填充到cls中
                    img_cls_counter = Counter(cls[i]).items()               #对每一图像中识别到的类型进行数量统计
                    img_cls_counter = sorted(img_cls_counter, key=lambda x: x[0])
                    #检测结果字典
                    detection_dict = {
                        '图片序号': i + 1,
                        '图片文件名': img_filename,
                        '检测结果': {},
                        '检测总数':{}
                    }
                    for cls_id, count in img_cls_counter:
                        cls_name = result.names[cls_id]
                        total_count += count
                        #写入CSV文件
                        writer.writerow({
                            '图片序号':i+1,
                            '图片文件名':img_filename,
                            '识别类型':cls_name,
                            '检测数量':count
                        })
                        #把每次检测出的结果再写进detection_dict中的检测结果字典
                        detection_dict['检测结果'][cls_name] = count
                        detection_dict['检测总数'][i + 1] = total_count
                else:
                    print(f"-----------第{i+1}张图片没有检测结果---------------")