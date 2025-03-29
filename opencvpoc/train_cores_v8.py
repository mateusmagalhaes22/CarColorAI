from ultralytics import YOLO

def main():
    model = YOLO("./runs/detect/train18/weights/best.pt")

    model.train(data="cores.yaml", epochs=60, batch=4, device=0)

if __name__ == '__main__':
    main()