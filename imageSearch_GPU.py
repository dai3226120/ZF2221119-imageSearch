from tqdm import tqdm
import glob
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# imgpixel = 512
# model = models.resnet18(pretrained=True)
# model = models.resnet34(pretrained=True)

imgpixel = 2048
# model = models.resnet50(pretrained=True)
# model = models.resnet101(pretrained=True)
model = models.resnet152(pretrained=True)

# imgpixel = 94080
# model = models.densenet201(pretrained=True)

# imgpixel = 1024
# model = models.googlenet(pretrained=True)


model = nn.Sequential(*list(model.children())[:-1])


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


    # cnn_feat = []
    # for path in tqdm(jpgs[:10]):
    #     img = cv2.imread(path)
    #     feat = cnn_feat_method(img)
    #     cnn_feat.append(feat.flatten())
    img_array = []                                      #装入所有训练图片
    img_array_test = []                                 #装入所有测试图片

    #训练图片处理
    for path in tqdm(jpgs[:]):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        img_array.append(img)

    img_array = np.stack(img_array)
    img_array = np.transpose(img_array, [0, 3, 1, 2]).astype(np.float32)
    img_array = torch.from_numpy(img_array)


    #测试图片处理
    for path in tqdm(jpgs_test[:]):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        img_array_test.append(img)

    img_array_test = np.stack(img_array_test)
    img_array_test = np.transpose(img_array_test, [0, 3, 1, 2]).astype(np.float32)
    img_array_test = torch.from_numpy(img_array_test)

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