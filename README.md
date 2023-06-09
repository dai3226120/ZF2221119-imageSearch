# ZF2221119-imageSearch



**解压缩img.rar和test.rar到项目根目录**

​	——img		训练集

​	——test		测试集



**imageSearch_GPU.py**

​	模型选择通过注释方式调整

​	包含：resnet18、resnet34、resnet50、resnet101、resnet152、densenet201和googlenet



​	使用不同的网络需要使用对应的imgpixel，包括：

​		512【resnet18、resnet34】

​		2048【resnet50、resnet101、resnet152】

​		94080【densenet201】

​		1024【googlenet】



​	search_idx表示测试集图片序号，单个数值可以取【0-99】



**imageSearch_vit_GPU.py**

​	模型选择通过注释方式调整

​	包含：vit_b_16、vit_b_32、vit_l_16、vit_l_32



​	search_idx表示测试集图片序号，单个数值可以取【0-99】



**需要具备的包**

tqdm

glob

sklearn.preprocessing

matplotlib.pyplot

numpy

cv2

torch

torch.nn

torchvision



**本机运行环境参考**

absl-py                 0.15.0
aiohttp                 3.8.1
aiosignal               1.2.0
async-timeout           4.0.2
attrs                   21.4.0
blinker                 1.4
brotlipy                0.7.0
cachetools              4.2.2
certifi                 2023.5.7
cffi                    1.15.1
charset-normalizer      2.0.4
click                   8.0.4
colorama                0.4.5
cryptography            37.0.1
cycler                  0.11.0
ffmpeg-python           0.2.0
fonttools               4.25.0
frozenlist              1.2.0
future                  0.18.2
google-auth             2.6.0
google-auth-oauthlib    0.4.4
grpcio                  1.42.0
idna                    3.3
imageio                 2.22.4
importlib-metadata      4.11.3
joblib                  1.1.1
kiwisolver              1.4.2
Markdown                3.3.4
matplotlib              3.5.2
mkl-fft                 1.3.1
mkl-random              1.2.2
mkl-service             2.4.0
multidict               6.0.2
munch                   2.5.0
munkres                 1.1.4
networkx                2.8.8
numpy                   1.24.3
oauthlib                3.2.1
opencv-python           4.6.0.66
packaging               21.3
pandas                  1.5.1
Pillow                  9.2.0
pip                     23.1.2
ply                     3.11
protobuf                3.20.1
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pycparser               2.21
PyJWT                   2.4.0
pyOpenSSL               22.0.0
pyparsing               3.0.9
PyQt5                   5.15.7
PyQt5-Qt5               5.15.2
PyQt5-sip               12.11.0
PyQtWebEngine           5.15.6
PyQtWebEngine-Qt5       5.15.2
PySocks                 1.7.1
python-dateutil         2.8.2
pytz                    2022.5
PyWavelets              1.4.1
QtPy                    2.2.0
requests                2.28.1
requests-oauthlib       1.3.0
rsa                     4.7.2
scikit-image            0.19.3
scikit-learn            1.2.0
scipy                   1.10.1
setuptools              63.4.1
sip                     6.6.2
six                     1.16.0
tensorboard             2.10.1
tensorboard-data-server 0.6.0
tensorboard-plugin-wit  1.8.1
threadpoolctl           2.2.0
tifffile                2022.10.10
toml                    0.10.2
torch                   1.12.1
torch-tb-profiler       0.4.0
torchaudio              0.12.1
torchvision             0.13.1
tornado                 6.2
tqdm                    4.64.1
typing_extensions       4.3.0
urllib3                 1.26.11
Werkzeug                2.0.3
wheel                   0.37.1
win-inet-pton           1.1.0
wincertstore            0.2
yarl                    1.8.1
zipp                    3.8.0
