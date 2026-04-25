from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == "__main__":
    model = YOLO("/root/workspace/experiment/yolo/ultralytics/cfg/models/11/yolo11-dinov3-depth-seg.yaml")

    add_swanlab_callback(
        model,
        project="YOLO_DINO",
        experiment_name="yolo11-dinov3-depth-seg",
        description="YOLO11 segmentation with a local DINOv3 RGB backbone and aligned depth CNN bypass. With epochs=120, batch=16, optimizer=AdamW, lr0=0.001, lrf=0.01, dino_lr0=5e-5, and no freezing of the DINOv3 backbone.",
    )

    model.train(
        data="depth.yaml",
        imgsz=640,
        epochs=120,
        batch=16,
        device=0,
        workers=4,
        optimizer="AdamW",
        dino_freeze=False, #是否冻结DINOv3 backbone的权重
        lr0=0.001,
        lrf=0.01,  #最终学习率比例
        dino_lr0=5e-5, #DINOv3 backbone的初始学习率
        project="runs/dinov3",
        name="yolo11-dinov3",
        pretrained=False,
    )
