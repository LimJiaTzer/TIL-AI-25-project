# After you have your fine-tuned 'best_rtdetr.pt'
from ultralytics import YOLO

def main():
    model = YOLO(r'./best.pt')
    model.export(format='engine', dynamic=True, imgsz=1280)

if __name__ == "__main__":
    main()