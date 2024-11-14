# Official ECF-YOLOv7-tiny

Implementation of paper - "ECF-YOLOv7-tiny: Improving feature fusion and the receptive field for lightweight object detectors".
The code and models will be made available after the paper becomes publicly available.

## Performance 

MS COCO

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | batch 32 fps on Jetson Nano |
| :-- | :-: | :-: | :-: | :-: | :-: |
| YOLOv7-tiny | 416 | **35.2%** | **52.8%** | **37.3%** | 14.4 *fps* |
| SE-YOLOv7-tiny | 416 | **34.8%** | **52.5%** | **36.6%** | 13.8 *fps* |
| YOLOv9-t | 416 | **32.2%** | **46.5%** | **34.2%** | 7.1 *fps* |
| YOLOv10-n | 416 | **32.6%** | **47.2%** | **35.0%** | - *fps* |
| **ECF-YOLOv7-tiny** | 416 | **38.2%** | **56.3%** | **40.7%** | 8.8 *fps* |
| **ECF-YOLOv7-tiny-s** | 416 | **37.8%** | **56.0%** | **40.0%** | 9.3 *fps* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name ecf-yolov7-tiny -it -v your_coco_path/:/coco/ -v your_code_path/:/ecf-yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /ecf-yolov7-tiny
```

</details>

## Testing

``` shell
#test ecf-yolov7-tiny
python test.py --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights weights/ecf-yolov7-tiny.pt --name ecf-yolov7-tiny_416_val

#test ecf-yolov7-tiny-s
python test.py --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights weights/ecf-yolov7-tiny-s.pt --name ecf-yolov7-tiny-s_416_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.750
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training

``` shell
# train ecf-yolov7-tiny
python train.py --workers 8 --device 0 --batch-size 32 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/training/ecf-yolov7-tiny.yaml --weights '' --name ecf-yolov7-tiny --hyp data/hyp.scratch.tiny.yaml

# train ecf-yolov7-tiny-s
python train.py --workers 8 --device 0 --batch-size 32 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/training/ecf-yolov7-tiny-s.yaml --weights '' --name ecf-yolov7-tiny-s --hyp data/hyp.scratch.tiny.yaml
```

Multiple GPU training

``` shell
# train ecf-yolov7-tiny
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 16 --device 0,1 --sync-bn --batch-size 32 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/training/ecf-yolov7-tiny.yaml --weights '' --name ecf-yolov7-tiny --hyp data/hyp.scratch.tiny.yaml

# train ecf-yolov7-tiny-s
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 16 --device 0,1 --sync-bn --batch-size 32 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/training/ecf-yolov7-tiny-s.yaml --weights '' --name ecf-yolov7-tiny-s --hyp data/hyp.scratch.tiny.yaml
```

## Inference

On video:
``` shell
python detect.py --weights weights/ecf-yolov7-tiny-s.pt --conf 0.25 --img-size 416 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights weights/ecf-yolov7-tiny-s.pt --conf 0.25 --img-size 416 --source inference/images/horses.jpg
```

## Check BFLOPs
``` shell
pip install thop
python detect_flops.py --weights weights/ecf-yolov7-tiny-s.pt --img-size 416 --conf 0.25 --source inference/images/horses.jpg
```

## Check speed
``` shell
python test.py --task speed --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights weights/ecf-yolov7-tiny-s.pt --name ecf-yolov7-tiny-s_416_b32_val_500ep
```

</details>

Tested with: Python 3.7.13, Pytorch 1.11.0+cu116

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
