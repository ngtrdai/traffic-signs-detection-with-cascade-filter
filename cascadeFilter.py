import cv2 as cv
import numpy as np
from pathlib import Path


dataFolder = Path('dataset')
imgFolder = dataFolder / 'images'

scrHieuLenh = './dataset/HieuLenh.xml'
scrCam = './dataset/Cam.xml'
scrHieuLenh_Cam = './dataset/Cam_HieuLenh.xml'

testImages = list(imgFolder.glob('*.bmp'))
labelFolder = dataFolder / 'labels'
lablels = list(labelFolder.glob('*.txt'))
path=imgFolder.glob("*.bmp")

# Load model
# cascadeHieuLenh = cv.CascadeClassifier(scrHieuLenh)
# cascadeCam = cv.CascadeClassifier(scrCam)
cascadeHieuLenhCam = cv.CascadeClassifier(scrHieuLenh_Cam)

# Điều chỉnh các thông số.
alpha = 1.3 # Độ tương phản
beta = 5 # Độ sáng
scaleFactor =  1.05
minNeighbors = 6
minSize = (15,15)
maxSize = (110, 110)

def XuLyLabel():
    '''
        - Hàm xử lý các file label chứa tọa độ.
        - Đưa các dữ liệu trong file vào một Dictionary
        - Cấu trúc của Dictionary: {'TenFile': ['SoLuongBienBao',[ListViTriBienBao]]}
    '''
    viTriDung = {}
    
    for file in lablels:
        data = np.loadtxt(file, delimiter = ",")
        fileName = str(file).split('\\')[2].split('.')[0]
        try:
            if data[0][0]:
                viTriDung[fileName] = [len(data),data]
        except:
            viTriDung[fileName] = [1,data]
     
    return viTriDung

def TienXuLyAnh(file):
    img = cv.imread(str(file), cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return img, gray

def IoU(boxA, boxB):
    # Tìm vị trí X,y phần diện tích A, B giao nhau
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# Diện tích phần chung của A và B
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
    #Diện tích của phân A và Phân B
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    #% vị trí A giao B
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou
            
def NhanDien(modelCascade, viTriDung):
    '''
        - Hàm nhận diện biển báo và tính toán IoU.
        - Đầu vào: 
            + modelCascade - file XML của model sau khi train.
            + viTriDung - một Dictionary có cấu trúc: {'TenFile': ['SoLuongBienBao',[ListViTriBienBao]]}
        - Đầu ra: Chỉ số IoU của thư mục ảnh test và kết quả trên từng ảnh ở trong thư mục "results".
    '''
    print("Đang nhận diện và tính IoU...")
    IoUs = []
    for file in path:
        # Tiền xử lý ảnh
        img, gray = TienXuLyAnh(file)
        fileName = str(file).split('\\')[2].split('.')[0]
        # Nhận diện biển báo Cấm và Hiệu lệnh với model được train gộp cả 2 biển
        NhanDienBienBao = modelCascade.detectMultiScale(gray,
                                                              scaleFactor=scaleFactor,
                                                              minNeighbors=minNeighbors,
                                                              minSize=minSize, 
                                                              maxSize = maxSize,
                                                              flags=cv.CASCADE_SCALE_IMAGE)
        
        # Tính IoU
        for (x,y,w,h) in NhanDienBienBao:
            # Vẽ vị trí của biển báo nhận dạng được (Màu xanh)
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)

            viTri = viTriDung[fileName]
            viTriTam =[]
            iouTam = -1

            # Trường 1 có 1 biển báo đúng
            if viTri[0]==1 :
                for viTriKetQua in viTri:
                    # Tách Vị trí và số lượng biển báo
                    try:
                        if viTriKetQua > 1000:
                            print("Loi")
                    except:
                        xMin = int(viTriKetQua[0])
                        yMin = int(viTriKetQua[1])
                        xMax = int(viTriKetQua[2])
                        yMax = int(viTriKetQua[3])
                        iou = IoU([x,y,x+w,y+h],[xMin,yMin,xMax,yMax])               
                        
                        # Vẽ biển báo đúng (Màu đỏ)
                        cv.rectangle(img,(xMin,yMin),(xMax,yMax),(0,0,255),1)
                        
                        # Hiển thị giá trị IoU cho từng biển báo 
                        iou = iou*100
                        cv.putText(img, "IoU = "+ "{:.2f}".format(iou)+"%", (x,y), cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0,255,0), thickness =2)
            
            # Trường họp có n biển báo đúng
            else :
                for danhSachViTriDung in viTri:
                    try:
                        # Tách Vị trí và số lượng biển báo
                        if danhSachViTriDung > 1000:
                            print("Loi")
                    except:
                        for viTriMoiBien in danhSachViTriDung:
                            xMin = int(viTriMoiBien[0])
                            yMin = int(viTriMoiBien[1])
                            xMax = int(viTriMoiBien[2])
                            yMax = int(viTriMoiBien[3])
                            iou = IoU([x,y,x+w,y+h],[xMin,yMin,xMax,yMax])  
                            if iou > iouTam:   
                                iouTam =iou
                                viTriTam = viTriMoiBien
                xMin = int(viTriTam[0])
                yMin = int(viTriTam[1])
                xMax = int(viTriTam[2])
                yMax = int(viTriTam[3])
                
                # Vẽ biển báo đúng (Màu đỏ)
                cv.rectangle(img,(xMin,yMin),(xMax,yMax),(0,0,255),1)
                #show IoU
                iou = iouTam*100
                cv.putText(img, "IoU = "+ "{:.2f}".format(iou)+"%", (x,y), cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0,255,0), thickness =2)
            IoUs.append(iou)
        cv.imwrite("./data/results/" + fileName + ".jpg", img)
    return np.mean(IoUs)

def main():
    viTriDung = XuLyLabel()
    ketQua = NhanDien(modelCascade=cascadeHieuLenhCam, viTriDung=viTriDung)
    print(f"IoU = {ketQua} %")

if __name__ == "__main__":
    main()
