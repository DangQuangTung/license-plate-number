import math
import cv2
import numpy as np
import Preprocess

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

#Đọc và thay đổi kích thước hình ảnh
img = cv2.imread("data/image/1.jpg")
img = cv2.resize(img, dsize=(1920, 1080))

#Tải mô hình KNN đã huấn luyện
npaClassifications = np.loadtxt("classifications.txt", np.float32) #Chứa các nhãn (label) cho từng ký tự.
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) #Chứa các hình ảnh ký tự đã được flatten (biến đổi thành mảng 1 chiều).
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # định hình lại mảng numpy thành 1d, cần thiết để chuyển sang lệnh gọi huấn luyện
kNearest = cv2.ml.KNearest_create()  # khởi tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
#Tải dữ liệu huấn luyện và tạo mô hình KNN từ các tệp đã lưu.

#Xử lý tiền đề hình ảnh
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img) # Chuyển đổi hình ảnh đầu vào sang ảnh xám bằng cách gọi hàm preprocess
canny_image = cv2.Canny(imgThreshplate, 250, 255) # Sử dụng thuật toán Canny để phát hiện cạnh trên ảnh đã được ngưỡng hóa
kernel = np.ones((3, 3), np.uint8) # tạo một kernel 3x3 (ma trận vuông) bằng cách sử dụng hàm np.ones
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Mở rộng các cạnh bằng phép giãn nở để làm nổi bật các vùng biên
#Tham số iterations=1 chỉ định rằng quá trình giãn nở sẽ được thực hiện một lần

###########################################

#Tìm các contour và lọc vùng biển số xe
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Tìm đường viền
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
# cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các contour trong hình lớn

screenCnt = [] #Lưu các contour có 4 cạnh (ứng với biển số xe).
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h

    if (len(approx) == 4):
        screenCnt.append(approx)

        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:  #Kiểm tra nếu đã phát hiện được biển số (detected == 1).

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Tìm góc của biển số xe #####################
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2) # Tính chiều cao (doi) và chiều rộng (ke) giữa hai điểm đã chọn.
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        

        #Cắt biển số xe và căn chỉnh cho vuông góc

        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
     

        # Cắt xén
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy] # vùng chứa biển số sau khi đã được cắt và xoay.
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        ####################################

        # Xử lý tiền đề và phân đoạn ký tự
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

        ##Lọc và nhận dạng ký tự
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h


            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind



        ## Nhận dạng ký tự

        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]  # Cắt các ký tự


            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # thay đổi kích thước hình ảnh
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # gọi hàm KNN find_nearest;
            strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII của ký tự
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            if (y < height / 3):  # quyết định biển số 1 hay 2 vạch
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))


        n = n + 1

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img) #Hiển thị kết quả

cv2.waitKey(0)
