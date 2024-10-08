# GenData.py

import numpy as np
import cv2
import sys


# biến cấp độ mô-đun
MIN_CONTOUR_AREA = 40


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")            # đọc trong hình ảnh số đào tạo
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize = None, fx = 0.5, fy = 0.5)
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # lấy hình ảnh thang độ xám
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # lờ mờ

                                                        # lọc ảnh từ thang độ xám sang đen trắng
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # hình ảnh đầu vào
                                      255,                                  # làm cho các pixel vượt qua ngưỡng có màu trắng hoàn toàn
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # sử dụng gaussian thay vì trung bình, dường như cho kết quả tốt hơn
                                      cv2.THRESH_BINARY_INV,                # đảo ngược để nền trước sẽ có màu trắng, nền sẽ có màu đen
                                      11,                                   # kích thước của vùng lân cận pixel được sử dụng để tính giá trị ngưỡng
                                      2)                                    # hằng số được trừ khỏi giá trị trung bình hoặc giá trị trung bình có trọng số

    cv2.imshow("imgThresh", imgThresh)      # hiển thị hình ảnh ngưỡng để tham khảo

    imgThreshCopy = imgThresh.copy()        # tạo một bản sao của hình ảnh đập lúa, điều này là cần thiết vì findContours sẽ sửa đổi hình ảnh

    npaContours, hierarchy = cv2.findContours(imgThreshCopy,        # hình ảnh đầu vào, hãy đảm bảo sử dụng một bản sao vì chức năng sẽ sửa đổi hình ảnh này trong quá trình tìm đường viền
                                                 cv2.RETR_EXTERNAL,                 # chỉ lấy các đường viền ngoài cùng
                                                 cv2.CHAIN_APPROX_SIMPLE)           # nén các đoạn ngang, dọc và chéo và chỉ để lại điểm cuối của chúng

                                # khai báo mảng có nhiều mảng trống, chúng ta sẽ sử dụng mảng này để ghi vào tập tin sau
                                # không có hàng nào, đủ cột để chứa tất cả dữ liệu hình ảnh
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
   

    intClassifications = []         # khai báo danh sách phân loại trống, đây sẽ là danh sách về cách chúng tôi phân loại ký tự từ đầu vào của người dùng, chúng tôi sẽ ghi vào tệp ở cuối

                                    # các ký tự có thể chúng ta quan tâm là các chữ số từ 0 đến 9, hãy đặt chúng vào danh sách intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')] #Là mã ascii của mấy chữ này

    for npaContour in npaContours:                          # cho mỗi đường viền
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # nếu đường viền đủ lớn để xem xét
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # nhận và thoát ra giới hạn chỉnh lưu

                                                # vẽ hình chữ nhật xung quanh mỗi đường viền khi chúng tôi yêu cầu người dùng nhập thông tin
            cv2.rectangle(imgTrainingNumbers,           # vẽ hình chữ nhật trên hình ảnh đào tạo ban đầu
                          (intX, intY),                 # góc trên bên trái
                          (intX+intW,intY+intH),        # góc dưới bên phải
                          (0, 0, 255),                  # màu đỏ
                          2)                            # độ dày

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # cắt char ra khỏi hình ảnh ngưỡng
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # thay đổi kích thước hình ảnh, điều này sẽ nhất quán hơn cho việc nhận dạng và lưu trữ

            cv2.imshow("imgROI", imgROI)                    # hiển thị char đã cắt ra để tham khảo
            cv2.imshow("imgROIResized", imgROIResized)      # hiển thị hình ảnh đã thay đổi kích thước để tham khảo
            
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # hiển thị hình ảnh số huấn luyện, bây giờ sẽ có hình chữ nhật màu đỏ được vẽ trên đó

            intChar = cv2.waitKey(0)                   

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # thoát khỏi chương trình
            elif intChar in intValidChars:      # khác nếu char nằm trong danh sách các ký tự mà chúng tôi đang tìm kiếm. . .

                intClassifications.append(intChar)        # nối char phân loại vào danh sách ký tự số nguyên (chúng tôi sẽ chuyển đổi thành float sau trước khi ghi vào tệp)
                #Là file chứa label của tất cả các ảnh mẫu, tổng cộng có 32 x 5 = 160 mẫu.
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # làm phẳng hình ảnh thành mảng numpy 1d để chúng ta có thể ghi vào tệp sau
                
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # thêm mảng có hình ảnh phẳng hiện tại vào danh sách các mảng có hình ảnh phẳng
                
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # chuyển đổi danh sách phân loại của int thành mảng float
    
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # làm phẳng mảng float thành 1d để chúng ta có thể ghi vào tệp sau

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # ghi hình ảnh phẳng vào tập tin
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # xóa windows khỏi bộ nhớ

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if
