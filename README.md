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
