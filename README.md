# YOLOv9 with ONNX & ONNXRuntime

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

This code providing two mode of inference, image and video inference. Basically, you just add `--image` flag for image inference and `--video` flag for video inference when running the python script.

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