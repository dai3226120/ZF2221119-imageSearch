from tqdm import tqdm
import glob
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


imgpixel = 1000
# model = models.vit_b_16(pretrained=True)
# model = models.vit_b_32(pretrained=True)
model = models.vit_l_16(pretrained=True)
# model = models.vit_l_32(pretrained=True)



def show_img(ori_path, aim_path, idx_):
    plt.figure(figsize=(20, 10))
    img = cv2.imread(ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3, 5, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')

    for idx in range(10):
        img = cv2.imread(aim_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 5, 6 + idx)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(idx_[idx])
    plt.show()

def cnn_feat_method(img):
    input = cv2.resize(img, (224, 224))             #标准化图形尺寸
    input = np.expand_dims(img, 0)                  #按axis=0扩展
    input = np.transpose(input, [0, 3, 1, 2])
    input = input.astype(np.float32)
    input = torch.from_numpy(input)

    with torch.no_grad():
        feat = model(input)

    feat = feat.flatten()
    feat = feat.data.numpy()
    return feat
def main():
    jpgs = glob.glob("./img/*.JPEG")                    #训练图片地址
    jpgs_test = glob.glob("./test/*.JPEG")                    #训练图片地址


    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(224),  # 将图像调整为模型期望的大小
        transforms.ToTensor(),   # 转换为张量
        transforms.Normalize(   # 标准化图像像素值
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_array = []                                      #装入所有训练图片
    img_array_test = []                                 #装入所有测试图片

    #训练图片处理
    for path in tqdm(jpgs[:]):
        # 加载图像
        img = Image.open(path).resize((224, 224)).convert('RGB')
        # 图像预处理
        input_tensor = preprocess(img)

        img_array.append(input_tensor)

    # 将输入张量堆叠成一个批量张量
    img_array = torch.stack(img_array)


    #测试图片处理
    for path in tqdm(jpgs_test[:]):
        # 加载图像
        img = Image.open(path).resize((224, 224)).convert('RGB')
        # 图像预处理
        input_tensor = preprocess(img)

        img_array_test.append(input_tensor)

    # 将输入张量堆叠成一个批量张量
    img_array_test = torch.stack(img_array_test)

#——————————————————————————————————————————————————————————————————————————————————————————————————#
    cnn_feat = []                                       #训练图片结果
    cnn_feat_test = []                                  #测试图片结果

    step = int(len(img_array) / 50) + 1                 #50个为一步

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_array = img_array.to(device)
    img_array_test = img_array_test.to(device)


    #装入训练图片结果
    print("装入训练图片结果:")
    for i in range(0, len(img_array), 50):
        # print(img_array[i: i + 50])
        with torch.no_grad():
            feat = model(img_array[i: i + 50].to(device))
            feat = feat.data.cpu().numpy()
            cnn_feat.append(feat)

        print(i, i / 50)

    cnn_feat = np.concatenate(cnn_feat, 0)
    cnn_feat = cnn_feat.reshape(-1, imgpixel)
    cnn_feat = normalize(cnn_feat)



    #装入测试图片结果
    print("装入测试图片结果:")
    for i in range(0, len(img_array_test), 50):
        # print(img_array[i: i + 50])
        with torch.no_grad():
            feat = model(img_array_test[i: i + 50].to(device))
            feat = feat.data.cpu().numpy()
            cnn_feat_test.append(feat)

        print(i, i / 50)

    cnn_feat_test = np.concatenate(cnn_feat_test, 0)
    cnn_feat_test = cnn_feat_test.reshape(-1, imgpixel)
    cnn_feat_test = normalize(cnn_feat_test)
#——————————————————————————————————————————————————————————————————————————————————————————————————#

    search_idx = [5,11,23,36,42,55,63,78,89]             #0-99

    for n in range(len(search_idx)):
        ids = np.dot(cnn_feat_test[search_idx[n]], cnn_feat.T)
        ids_imgid = np.argsort(ids)[::-1][1:]
        show_img(jpgs_test[search_idx[n]], [jpgs[x] for x in ids_imgid[:10]], [ids[x] for x in ids_imgid[:10]])

    
if __name__ == '__main__':
    main()

