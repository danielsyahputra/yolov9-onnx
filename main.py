import os
import cv2
from pathlib import Path

from yolov9 import YOLOv9


def get_detector(args):
    weights_path = args.weights
    classes_path = args.classes
    source_path = args.source
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

    if args.image:
        image = cv2.imread(source_path)
        h,w = image.shape[:2]
    elif args.video:
        cap = cv2.VideoCapture(source_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = YOLOv9(model_path=weights_path,
                      class_mapping_path=classes_path,
                      original_size=(w, h),
                      score_threshold=args.score_threshold,
                      conf_thresold=args.conf_threshold,
                      iou_threshold=args.iou_threshold,
                      device=args.device)
    return detector

def inference_on_image(args):
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    image = cv2.imread(args.source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)

    output_path = f"output/{Path(args.source).name}"
    print(f"[INFO] Saving result on {output_path}")
    cv2.imwrite(output_path, image)

    if args.show:
        cv2.imshow("Result", image)
        cv2.waitKey(0)

def inference_on_video(args):
    print("[INFO] Intialize Model")
    detector = get_detector(args)

    cap = cv2.VideoCapture(args.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter('output/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))

    print("[INFO] Inference on Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        detector.draw_detections(frame, detections=detections)
        writer.write(frame)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] Finish. Saving result to output/result.avi")

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Argument for YOLOv9 Inference using ONNXRuntime")

    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--weights", type=str, required=True, help="Path to yolov9 onnx file")
    parser.add_argument("--classes", type=str, required=True, help="Path to list of class in yaml file")
    parser.add_argument("--score-threshold", type=float, required=False, default=0.1)
    parser.add_argument("--conf-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--iou-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--image", action="store_true", required=False)
    parser.add_argument("--video", action="store_true", required=False)
    parser.add_argument("--show", required=False, type=bool, default=True, help="Show result on pop-up window")
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cude)", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    if args.image:
        inference_on_image(args=args)
    elif args.video:
        inference_on_video(args=args)
    else:
        raise ValueError("You can't process the result because you have not define the source type (video or image) in the argument")
    