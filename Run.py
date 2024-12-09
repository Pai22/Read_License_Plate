from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def plot_license_plate(results, model):
    all_boxes = []  # เก็บพิกัดของทุกกรอบ
    for result in results:
        for box in result.boxes:
            # พิกัดกรอบในรูปแบบ [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # ดึงชื่อคลาสจาก index
            class_index = int(box.cls[0])  # ต้องแปลงเป็น int ก่อน
            class_name = model.names[class_index]  # ดึงชื่อคลาส
            
            # พิมพ์ความมั่นใจและชื่อคลาส
            print(f"Confidence: {box.conf[0]:.2f}, Class Name: {class_name}")
            
            # เก็บพิกัดเป็น int ใน all_boxes
            all_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        result_img = result.plot()
        plt.figure(1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Main photo')
        plt.axis('off')
    
    return all_boxes  # ส่งคืนพิกัดของทุกกรอบ

def read_license_plate(results, model,i):
    for result in results:
        print(f"License Plate {i}")
        for box in result.boxes:            
            # ดึงชื่อคลาสจาก index
            class_index = int(box.cls[0])  # ต้องแปลงเป็น int ก่อน
            class_name = model.names[class_index]  # ดึงชื่อคลาส
            
            # พิมพ์ความมั่นใจและชื่อคลาส
            print(f"Confidence: {box.conf[0]:.2f}, Class Name: {class_name}")
            

# โหลดโมเดล
Crop_License_Plate_model = YOLO("train_Crop_License_Plate/train/weights/best.pt")
Read_License_Plate_model = YOLO("train_Read_License_Plate/train/weights/best.pt")

# อ่านภาพต้นฉบับ
image_path = "CropPlate/20220223194124-7a9f_wm_jpg.rf.1d7ad48b122e5ca57f6f24d8a2467551.jpg"
original_image = cv2.imread(image_path)

# ตรวจจับกรอบป้ายทะเบียน
Crop_results = Crop_License_Plate_model.predict(image_path)
all_boxes = plot_license_plate(Crop_results, Crop_License_Plate_model)

# อ่านข้อมูลจากกรอบที่ครอบแต่ละกรอบ
for i, (x1, y1, x2, y2) in enumerate(all_boxes, start=2):  # เริ่มที่ Figure 2
    cropped_image = original_image[y1:y2, x1:x2]  # ตัดภาพตามพิกัด
    Read_result = Read_License_Plate_model.predict(cropped_image)
    read_license_plate(Read_result, Read_License_Plate_model, i)
    plt.figure(i)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title(f'License Plate {i}')
    plt.axis('off')

# แสดงผลทั้งหมด
plt.show()