from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-dinov3.yaml")
    # For segmentation, use:
    # model = YOLO("ultralytics/cfg/models/11/yolo11-dinov3-seg.yaml")

    add_swanlab_callback(
        model,
        project="YOLO11-DINOv3",
        experiment_name="yolo11-dinov3",
        description="YOLO11 with a local DINOv3 backbone.",
    )

    model.train(
        data="your_dataset.yaml",
        imgsz=640,
        epochs=100,
        batch=8,
        device=0,
        workers=4,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,  #最终学习率比例
        dino_lr0=5e-5, #DINOv3 backbone的初始学习率
        project="runs/dinov3",
        name="yolo11-dinov3",
        pretrained=False,
    )
