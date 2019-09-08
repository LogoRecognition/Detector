# Detector for Brand Logo Dataset

This is based on the implementation of Faster-RCNN(https://github.com/smallcorgi/Faster-RCNN_TF). In LogoReader application, we will take Faster-RCNN as detector to find out the logos in our logo dataset. In this part, we will provide an easy-to-use interface to take an image in and output the information about the logo in the image.


## Training and Testing

python ./tools/train_net.py --device gpu --device_id 0 --solver VGG_CNN_M_1024 --weight ./data/pretrain_model/VGG_imagenet.npy --imdb carlogo_27_train --network calogo27_train

python ./tools/train_net.py --device gpu --device_id 0 --solver VGG_CNN_M_1024 --weight ./data/pretrain_model/VGG_imagenet.npy --imdb carlogo_27_test --network calogo27_test

## Detector Interface
### environment
source ~/.bashrc

source activate tensorflow

### if something wrong with Display
(tensorflow) root@LogoRecognitionVM:~/Faster-RCNN_TF# vncserver

New 'LogoRecognitionVM:4 (root)' desktop is LogoRecognitionVM:4

Starting applications specified in /root/.vnc/xstartup
Log file is /root/.vnc/LogoRecognitionVM:4.log

(tensorflow) root@LogoRecognitionVM:~/Faster-RCNN_TF# export DISPLAY=LogoRecognitionVM:4

(tensorflow) root@LogoRecognitionVM:~/Faster-RCNN_TF# xhost +
### file structure
define an extractor class，which provides a get_feature interface


### how to use
cd /home/workplace

python

from Faster\_RCNN\_TF.tools.extract_features import extractor
e = extractor()

e.get_feature(图片名字)

### input

images are saved in
./Detector/data/demo/

e.g. ./Detector/data/demo/0075.jpg
the input format should be '0075.jpg'

### output
output images will be saved in
./detect/

### if there is nothing detected 
when you input '0075.jpg', if there is a logo, there will be an output image in ./detect/0075.jpg, otherwise not.

## Brand Logo Dataset
we collect 27 brand classes images, up to 4360.
30% of each brand's images are taken as test set.
Brand coverage: Car, Sports, Drinks, Techs, Fast Food, E-commerce

| Classes | total | train | test |
|---------|-------|-------|------|
|Acura|260|78|182|
|Alpha-Romeo|234|71|163|
|Aston-Martin|280|84|196|
|Audi|229|69|160|
|Bentley|302|91|211|
|Benz|301|90|211|
|BMW|276|83|193|
|Bugatti|200|60|140|
|Buick|273|82|191|
|Nike|192|58|134|
|Adidas|230|69|161|
|Vans|263|79|184|
|Converse|222|67|155|
|Puma|302|91|211|
|New Balance|195|59|136|
|Anta|208|63|145|
|Lining|204|62|142|
|Pessi|205|62|143|
|Yili|203|62|141|
|Uniqlo|200|61|139|
|Coca|202|61|141|
|Haier|195|59|136|
|Huawei|220|66|154|
|Apple|224|68|156|
|Lenovo|200|60|140|
|McDonalds|215|65|150|
|Amazon|210|63|147|



### The result of testing on Brand Logo Dataset

| Classes       | AP     |
|-------------|--------|
|'__background__'|0.811|
|Acura|0.961|
|Alpha-Romeo|0.993|
|Aston-Martin|0.696|
|Audi|0.902|
|Bentley|0.856|
|Benz|0.899|
|BMW|0.828|
|Bugatti|0.715|
|Buick|0.604|
|Nike|0.784|
|Adidas|0.604
|Vans|0.663|
|Converse|0.51|
|Puma|0.591|
|New Balance|0.545|
|Anta|0.331|
|Lining|0.688|
|Pessi|0.68|
|Yili|0.731|
|Uniqlo|0.443|
|Coca|0.982|
|Haier|0.897|
|Huawei|0.566|
|Apple|0.977|
|Lenovo|0.93|
|McDonalds|0.977|
|Amazon|0.747|
| mAP        | 0.747|


###References
[Faster R-CNN TensorFlow version](https://github.com/smallcorgi/Faster-RCNN_TF)

