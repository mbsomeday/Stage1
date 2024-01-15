import json
import os
import cv2
from tqdm import tqdm


JSON_DIR = r'D:\chrom_download\ECP\ECP_day_labels_val\ECP\day\labels\val'
IMG_DIR = r'D:\chrom_download\ECP\ECP_day_img_val\ECP\day\img\val'
GROUPED_ECP = r'D:\chrom_download\grouped_ECP'


def seg_object(image_path, x0, x1, y0, y1, cropped_name):
    image = cv2.imread(image_path)
    cropped = image[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    # image = cv2.rectangle(image, (x1, y1), (x0, y0), (0, 0, 255), 2) # (右下角), (左上角)
    # cv2.imshow('image', image)
    # key = cv2.waitKey(0)
    cv2.imwrite(cropped_name, cropped)


def read_json(json_file, image_path):
    with open(json_file, mode='r', encoding='utf-8') as fp:
        user_dic = json.load(fp=fp)
        children = user_dic['children']
        # print('number of object in this image:', len(children))
        for i in range(len(children)):
            info = children[i]
            identity = info['identity']
            # print('current class:', identity)
            x0 = info['x0']
            x1 = info['x1']
            y0 = info['y0']
            y1 = info['y1']

            new_class_dir = os.path.join(GROUPED_ECP, identity)
            # 若该类别不存在，则创建
            object_idx = 0
            if not os.path.exists(new_class_dir):
                os.mkdir(new_class_dir)
            else:
                object_idx = len(os.listdir(new_class_dir))
            # 根据坐标裁剪图片
            cropped_name = os.path.join(GROUPED_ECP, identity, str(object_idx) + '.png')

            seg_object(image_path, x0, x1, y0, y1, cropped_name)

            # break


label_dir_list = os.listdir(JSON_DIR)

for label_dir in label_dir_list:
    print("current label:", label_dir)
    curent_dir = os.path.join(JSON_DIR, label_dir)
    json_files = os.listdir(curent_dir)

    for json_file in tqdm(json_files):
        file_name = json_file[:-5]
        image_path = os.path.join(IMG_DIR, label_dir, file_name + '.png')
        json_path = os.path.join(JSON_DIR, label_dir, json_file)
        read_json(json_path, image_path)












