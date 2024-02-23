# YOLOv9 with ONNX & ONNXRuntime

Performing Object Detection for YOLOv9 with ONNX and ONNXRuntime

![! ONNX YOLOv9 Object Detection](https://github.com/danielsyahputra/yolov9-onnx/blob/master/output/sample_image.jpeg)


## Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Installation

```shell
git clone https://github.com/danielsyahputra/yolov9-onnx.git
cd yolov9-onnx
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

## ONNX model and Class metadata

You can download the onnx model and class metadata file on the link below

```
https://drive.google.com/drive/folders/1QH5RCF5WOk53SfdzsHTFkXAdzMLbbQeO?usp=sharing
```

## Examples

### Arguments
List  the arguments available in main.py file.

- `--source`: Path to image or video file
- `--weights`: Path to yolov9 onnx file (ex: weights/yolov9-c.onnx)
- `--classes`: Path to yaml file that contains the list of class from model (ex: weights/metadata.yaml)
- `--score-threshold`: Score threshold for inference, range from 0 - 1
- `--conf-threshold`: Confidence threshold for inference, range from 0 - 1
- `--iou-threshold`: IOU threshold for inference, range from 0 - 1
- `--image`: Image inference mode
- `--video`: Video inference mode
- `--show`: Show result on pop-up window
- `--device`: Device use for inference, default = cpu.



Note: If you want to use `cuda` for inference, please make sure you are already install `onnxruntime-gpu` before running the script.


This code provides two modes of inference, image and video inference. Basically, you just add `--image` flag for image inference and `--video` flag for video inference when you are running the python script.


If you have your own custom model, don't forget to provide a yaml file that consists the list of class that your model want to predict. This is example of yaml content for defining your own classes:

```
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  .
  .
  .
  .
  n: object
```

### Inference on Image

```
python main.py --source assets/sample_image.jpeg --weights weights/yolov9-c.onnx --classes weights/metadata.yaml --image
```

### Inference on Video

```
python main.py --source assets/road.mp4 --weights weights/yolov9-c.onnx --classes weights/metadata.yaml --video
```

# References:
* YOLOv9 model: [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)