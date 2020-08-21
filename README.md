# DesFPN-detectron2
detection

训练
```
!python \
DesFPN-detectron/train_net.py \
--config-file DesFPN-detectron/configs/COCO-Detection/faster_rcnn_DesFPN_1x.yaml \
MODEL.WEIGHTS output/model_0009999.pth
```

测试
```
!python \
DesFPN-detectron/train_net.py \
--config-file DesFPN-detectron/configs/COCO-Detection/faster_rcnn_DesFPN_1x.yaml \
--eval-only MODEL.WEIGHTS output/model_0009999.pth
```
检测
```
!python \
DesFPN-detectron/demo.py \
--config-file DesFPN-detectron/configs/COCO-Detection/faster_rcnn_DesFPN_1x.yaml \
--input 236.jpg \
--output . \
--opts MODEL.WEIGHTS output/model_0009999.pth
```