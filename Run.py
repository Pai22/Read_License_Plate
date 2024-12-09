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
    
    return all_boxes  # 

def english_to_thai(class_name):
    """
    แปลงตัวอักษรภาษาอังกฤษ A01-A44 เป็นพยัญชนะภาษาไทย ก-ฮ
    """
    # สร้าง mapping ของพยัญชนะไทย
    thai_chars = [
        "ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", 
        "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", 
        "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
    ]
    
    # ตรวจสอบว่าคลาสเป็นรูปแบบ Axx
    if class_name.startswith("A") and len(class_name) > 1:
        try:
            index = int(class_name[1:]) - 1  # แปลง A01-A44 เป็น index
            if 0 <= index < len(thai_chars):
                return thai_chars[index]
        except ValueError:
            pass
    
    return class_name  # หากไม่ตรงเงื่อนไข คืนค่าดั้งเดิม

def province_abbreviation_to_name(abbreviation):
    """
    แปลงอักษรย่อของจังหวัดในประเทศไทยเป็นชื่อเต็ม
    """
    province_map = {
    "BKK": "กรุงเทพมหานคร",
    "KBI": "กระบี่",
    "KRI": "กาญจนบุรี",
    "KSN": "กาฬสินธุ์",
    "KPT": "กำแพงเพชร",
    "KKN": "ขอนแก่น",
    "CTI": "จันทบุรี",
    "CCO": "ฉะเชิงเทรา",
    "CBI": "ชลบุรี",
    "CNT": "ชัยนาท",
    "CPM": "ชัยภูมิ",
    "CPN": "ชุมพร",
    "CRI": "เชียงราย",
    "CMI": "เชียงใหม่",
    "TRG": "ตรัง",
    "TRT": "ตราด",
    "TAK": "ตาก",
    "NYK": "นครนายก",
    "NPT": "นครปฐม",
    "NPM": "นครพนม",
    "NMA": "นครราชสีมา",
    "NST": "นครศรีธรรมราช",
    "NSN": "นครสวรรค์",
    "NBI": "นนทบุรี",
    "NWT": "นราธิวาส",
    "NAN": "น่าน",
    "BKN": "บึงกาฬ",
    "BRM": "บุรีรัมย์",
    "PTE": "ปทุมธานี",
    "PKN": "ประจวบคีรีขันธ์",
    "PRI": "ปราจีนบุรี",
    "PTN": "ปัตตานี",
    "PYO": "พะเยา",
    "AYA": "พระนครศรีอยุธยา",
    "PNA": "พังงา",
    "PLG": "พัทลุง",
    "PCT": "พิจิตร",
    "PLK": "พิษณุโลก",
    "PBI": "เพชรบุรี",
    "PNB": "เพชรบูรณ์",
    "PRE": "แพร่",
    "PKT": "ภูเก็ต",
    "MKM": "มหาสารคาม",
    "MDH": "มุกดาหาร",
    "MSN": "แม่ฮ่องสอน",
    "YST": "ยโสธร",
    "YLA": "ยะลา",
    "RET": "ร้อยเอ็ด",
    "RNG": "ระนอง",
    "RYG": "ระยอง",
    "RBR": "ราชบุรี",
    "LRI": "ลพบุรี",
    "LPG": "ลำปาง",
    "LPN": "ลำพูน",
    "LEI": "เลย",
    "SSK": "ศรีสะเกษ",
    "SNK": "สกลนคร",
    "SKA": "สงขลา",
    "STN": "สตูล",
    "SPK": "สมุทรปราการ",
    "SKM": "สมุทรสงคราม",
    "SKN": "สมุทรสาคร",
    "SKW": "สระแก้ว",
    "SRI": "สระบุรี",
    "SBR": "สิงห์บุรี",
    "STI": "สุโขทัย",
    "SPB": "สุพรรณบุรี",
    "SNI": "สุราษฎร์ธานี",
    "SRN": "สุรินทร์",
    "NKI": "หนองคาย",
    "NBP": "หนองบัวลำภู",
    "ATG": "อ่างทอง",
    "ACR": "อำนาจเจริญ",
    "UDN": "อุดรธานี",
    "UTT": "อุตรดิตถ์",
    "UTI": "อุทัยธานี",
    "UBN": "อุบลราชธานี",
}

    return province_map.get(abbreviation, abbreviation)  # คืนค่าชื่อเต็มหรือคืนค่าเดิมหากไม่พบ



def read_license_plate(results, model, i):
    """
    อ่านตัวอักษรจากกรอบที่ครอบและแปลงอักษรย่อ/พยัญชนะเป็นภาษาไทย
    """
    xy = []  # ใช้ list เพื่อเก็บข้อมูล box ทั้งหมด
    for result in results:
        print(f"License Plate {i}")
        if len(result.boxes.xyxy) != 0:
            for box in result.boxes:
                # ดึงข้อมูลของ bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_index = int(box.cls[0])  # ดึง index ของ class
                class_name = model.names[class_index]  # ดึงชื่อคลาส
                confidence = float(box.conf[0])  # ความมั่นใจ
                
                # แปลงชื่อคลาส
                if class_name.isalpha():  # ตรวจสอบว่าเป็นตัวอักษรล้วน
                    thai_char = province_abbreviation_to_name(class_name)  # แปลงเป็นชื่อจังหวัด
                else:
                    thai_char = english_to_thai(class_name)  # แปลงเป็นพยัญชนะไทย
                
                # เพิ่มข้อมูล box ลงใน list
                xy.append({
                    "coordinates": [x1, y1, x2, y2],
                    "class_name": thai_char,
                    "confidence": confidence
                })
            
            # จัดเรียง box ตามค่า y1
            sorted_boxes = sorted(xy, key=lambda ob: ob["coordinates"][1])  # ใช้ y1
            provide = sorted_boxes[len(sorted_boxes)-1]
            sorted_boxes.pop(len(sorted_boxes)-1)
            final_sorted_boxes = sorted(sorted_boxes, key=lambda ob: ob["coordinates"][0])  # ใช้ x1
            
            # แสดงผลเรียงลำดับ
            for result in final_sorted_boxes:
                print(result["class_name"])
                
            print(provide["class_name"])
        else:
            print("Can't detect label in this License Plate")


# โหลดโมเดล
Crop_License_Plate_model = YOLO("train_Crop_License_Plate/train/weights/best.pt")
Read_License_Plate_model = YOLO("train_Read_License_Plate/train/weights/best.pt")

# อ่านภาพต้นฉบับ
image_path = "D:/Intren/Detect license plates/roboflow3/train/images/183-89-207-243_15_20220906150707927_jpg.rf.bedfcfed9466a7473cc9457c77e19a3e.jpg"
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
