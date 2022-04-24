#we gitpull from here so i am not sure i need to cite, but to be safe most of this comes from: https://github.com/postor/DOTA-yolov3

#cd /home/ubuntu/project/
#git clone https://github.com/postor/DOTA-yolov3.git
#pip install Shapely
#cd /home/ubuntu/project/DOTA-yolov3
#mkdir dataset
#cd /home/ubuntu/project/DOTA-yolov3/dataset
#mkdir train
#mkdir val
#cd /home/ubuntu/project/DOTA-yolov3/dataset/train
#mkdir labelTxt
#cd /home/ubuntu/project/DOTA-yolov3/dataset/val
#mkdir labelTxt
#cd
#cp /home/ubuntu/data/part1.zip /home/ubuntu/project/DOTA-yolov3/dataset/train/part1.zip
#cp /hom/ubuntu/data/part2.zip /home/ubuntu/project/DOTA-yolov3/dataset/train/part2.zip
#cp /hom/ubuntu/data/part3.zip /home/ubuntu/project/DOTA-yolov3/dataset/val/part3.zip
#cp /hom/ubuntu/data/DOTA-v1.5_train.zip /home/ubuntu/project/DOTA-yolov3/dataset/train/labelTxt/DOTA-v1.5_train.zip
#cp /hom/ubuntu/data/DOTA-v1.5_train.zip /home/ubuntu/project/DOTA-yolov3/dataset/val/labelTxt/DOTA-v1.5_train.zip
#cd /home/ubuntu/project/DOTA-yolov3/dataset/train
#unzip -u part1.zip
#unzip -u part2.zip
#unzip -u DOTA-v1.5_train.zip
#find . -size +5M -delete
#cd /home/ubuntu/project/DOTA-yolov3/dataset/val
#unzip -u part3.zip
#unzip -u DOTA-v1.5_train.zip
#find . -size +5M -delete
#cd /home/ubuntu/project/DOTA-yolov3
#python3 data_transform/split.py
#mkdir /home/ubuntu/project/DOTA-yolov3/dataset/trainsplit/labels
#mkdir /home/ubuntu/project/DOTA-yolov3/dataset/valsplit/labels
#python3 data_transform/YOLO_Transform.py
#ls -1d $PWD/dataset/trainsplit/images/* > cfg/train.txt
#ls -1d $PWD/dataset/valsplit/images/* > cfg/val.txt
#cd cfg
#mkdir backup
#open dota.data in notepad and change train = train.txt to train = /home/ubuntu/project/DOTA-yolov3/cfg/train.txt and  valid = test.txt to valid = /home/ubuntu/project/DOTA-yolov3/cfg/test.txt

#Install Darknet
#cd /home/ubuntu/project/
#git clone https://github.com/pjreddie/darknet.git
#cd darknet
#make
#Change First line of makefile to GPU=1
# now run it: ./darknet

#./darknet detector train /home/ubuntu/project/DOTA-yolov3/cfg/dota.data /home/ubuntu/project/DOTA-yolov3/cfg/dota-yolov3-tiny.cfg

