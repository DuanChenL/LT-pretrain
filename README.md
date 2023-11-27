# Long-tail Detection with Effective Class-Margins 


## Installation
### Requirements 

We tested our codebase on mmdet 2.25.3, mmcv 1.6.1, PyTorch 1.12.0, torchvision 0.13.0, and python 3.9. 

### Setup
To setup the code, please follow the commands below:

~~~

# Install mmcv.
pip install -U openmim
mim install mmcv-full==1.6.1

# And mmdetection. 
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .

# Additionally, install lvis-api. 
pip install lvis
~~~

### Dataset 
下载 [LVIS dataset](https://www.lvisdataset.org/dataset)，其结构如下所示. 
~~~
data
  ├── lvis_v1
  |   ├── annotations
  │   │   │   ├── lvis_v1_val.json
  │   │   │   ├── lvis_v1_train.json
  │   ├── train2017
  │   │   ├── 000000004134.png
  │   │   ├── 000000031817.png
  │   │   ├── ......
  │   ├── val2017
  │   ├── test2017
~~~
将"baseline\configs\_base_\datasets\lvis_v1_instance_ssl.py"中的data_root修改为对应地址即可

### Training
下载用于自监督训练的目标框文件，将其放入"mmdetection/mmdet/models/detectors"目录下：

链接：https://pan.baidu.com/s/1Z7j4y1bs7jjDhvIx33xV2g?pwd=g40e 

提取码：g40e 

--来自百度网盘超级会员V5的分享


resnet-50-FPN训练24 epochs命令：
~~~

./sh_files/r50_2x.sh 

~~~
