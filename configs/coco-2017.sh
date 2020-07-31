#!/bin/bash
# Zip coco folder
# zip -r coco.zip coco
# tar -czvf coco.tar.gz coco
mkdir datasets
cd datasets

# Download labels from Google Drive, accepting presented query
filename="coco2017labels.zip"
fileid="1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip labels
unzip -q ${filename}  # for coco.zip
# tar -xzf ${filename}  # for coco.tar.gz
rm ${filename}

# Download and unzip images
cd coco
f="train2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f
f="val2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f

cd annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cd annotations
mv instances_train2017.json ../
# cd out
cd ../../..

