#from daam import trace, set_seed
from daam.utils import set_seed
from daam.trace import trace
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch
import os
from collections import defaultdict
import urllib
import requests
import json
import cv2
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageOps
# def s_s(sentence,word):
#     sentence=sentence.split(' ')
#     sentence_flag=[]
#     for words in sentence:
#         sentence_flag.append(0)
#         len_=min(len(words),len(word)) 
#         for id in range(len_):
#             if words[id] == word[id]:
#                 sentence_flag[-1]+=1
#             else:
#                 break
#     x = np.array(sentence_flag).argmax()
#     return sentence[x]
def shenchuanguo_s(sentence, word):
    sentence_list = sentence.split(' ')
    sentence_flag = []
    for words in sentence_list:
        # 初始化矩阵，用于记录两个单词的编辑距离
        matrix = np.zeros((len(word) + 1, len(words) + 1))
        for i in range(len(word) + 1):
            matrix[i][0] = i
        for j in range(len(words) + 1):
            matrix[0][j] = j
        # 计算编辑距离
        for i in range(1, len(word) + 1):
            for j in range(1, len(words) + 1):
                if word[i - 1] == words[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(matrix[i][j - 1], matrix[i - 1][j], matrix[i - 1][j - 1]) + 1
        # 记录相似度
        sentence_flag.append(len(word) + len(words) - matrix[-1][-1])
    x = np.array(sentence_flag).argmax()
    return sentence_list[x]
def get_index(cococlass, caption):
    result = {}
    for word in cococlass:
        if word in caption:
            result[word] = caption.index(word)
    return result
model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token="？？？")
pipe = pipe.to(device)
gen = set_seed(15)  # for reproducibility3
annFile = 'coco/captions_train2017.json'
annFile1='coco/instances_train2017.json'#我们首先初始化了COCO API，然后遍历了所有标注。对于每个标注，我们获取其caption和image_id，并使用image_id查找对应的图像信息。然后，我们从图像信息中获取category_id，并使用category_id查找对应的category名称
coco = COCO(annFile)
coco1 = COCO(annFile1)
save_path = '100images/maps'
caption_count = 0 
if not os.path.exists(save_path):
    os.makedirs(save_path)
with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
     with trace(pipe) as tc:
          category_captions = defaultdict(list)
          for ann in coco.anns.values():
            caption = ann['caption']
            print(caption)
            
            img_id = ann['image_id']
            ann_ids = coco1.getAnnIds(imgIds=img_id)
            if not ann_ids:
              continue
            anns = coco1.loadAnns(ann_ids)
            category_id = anns[0]['category_id']
            category = coco1.loadCats(category_id)[0]['name']
            print(category )

            
            if category not in caption :
                  continue
        # 如果该category的captions数量已经达到1000，则跳过
            if len(category_captions[category]) > 100:
                 continue
            if caption_count >8000:    # 增加检查是否达到终止条件的逻辑
                 break
            if caption.endswith("."):
               caption = caption[:-1]
            category_captions[category].append(caption)
            caption_count += 1
            print('caption_count:',caption_count)
            out = pipe(caption, num_inference_steps=？, generator=gen)
            heat_map = tc.compute_global_heat_map()
            filename = category + '_' + str(ann['image_id']) + '_' + str(ann['id']) +  ".png"

            cococlass = ['background',
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            cocolist=list(cococlass)
            index1=cocolist.index(category)
            cococlass1= ['woman','man','boy','girl','child']
            cocolist1=list(cococlass1)
            caption_words = caption.split()
            fruit_list = []
            for i, word in enumerate(cocolist):
               if word in caption:
                  fruit = word
                  index = i
                  fruit_list.append((fruit, index))
            for i, word in enumerate(cocolist1):
               if word in caption_words:
                  fruit = word
                  index = 1
                  fruit_list.append((fruit, index))
            l= len(fruit_list)
            image = []
            for fruit,index in fruit_list:
                print(fruit,index)
                category_r = s_s(caption,fruit)
                print(category_r)
                heat_map1 = heat_map.compute_word_heat_map(category_r)
                heat_map1.plot_overlay(out.images[0])
                filename1 = category + '_' + str(ann['image_id']) + '_' + str(ann['id']) +  str(index)+ fruit +  ".png"
                plt.savefig(os.path.join(save_path, filename1),bbox_inches='tight', pad_inches=0)
                img = cv2.imread(os.path.join(save_path, filename1), cv2.IMREAD_GRAYSCALE)
                threshold_value=100
                ret, binary = cv2.threshold(img, threshold_value, index, cv2.THRESH_BINARY)
                # print(img.shape)
                image.append(binary)
                os.remove(os.path.join(save_path, filename1))
            merged_image = np.zeros_like(image[0])
            height, width = merged_image.shape[:2]
            print(merged_image.shape)
            for i in range(l):
            #  merged_image[np.logical_and(images[i] > 0, images[i] > merged_image)] = images[i][np.logical_and(images[i] > 0, images[i] > merged_image)]
             for row in range(369):
               for col in range(369):
                   if image[i][row][col] == index1:
                      merged_image[row][col] = index1
                   elif merged_image[row][col] != index1 and image[i][row][col] > merged_image[row][col]:
                    merged_image[row][col] = image[i][row][col]
            cv2.imwrite(os.path.join(save_path, filename), merged_image)
            # print(merged_image.shape)

            # else:
            #  print('Result is empty. Skipping...')
                # cv2.imwrite(os.path.join(save_path, filename), gray)