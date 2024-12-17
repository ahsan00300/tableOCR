import cv2
from ultralytics import YOLO
from ultralyticsplus import YOLO, render_result
import numpy as np

from tqdm import tqdm
import easyocr
from pdf2image import convert_from_path

reader = easyocr.Reader(['th','en'])

class ocr_table:

    def __init__(self):
        pass

    def pdf_to_images(self, pdf_path, output_folder=None):
        """
        Converts a PDF into a list of images (one per page).

        Args:
            pdf_path (str): Path to the input PDF file.
            output_folder (str, optional): Directory where images will be saved (optional).

        Returns:
            list: A list of PIL Image objects representing PDF pages.
        """
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            images = [np.array(image) for image in images]
            # # Save images to an output folder if provided
            # if output_folder:
            #     os.makedirs(output_folder, exist_ok=True)
            #     for i, img in enumerate(images):
            #         image_path = os.path.join(output_folder, f"page_{i + 1}.png")
            #         img.save(image_path, "PNG")
            
            return images
        
        except Exception as e:
            print(f"Error while processing PDF: {e}")
            return []

    def extract_table(self, image):

        # load model
        model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        model.overrides['conf'] = 0.25  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000  # maximum number of detections per image

        # perform inference
        results = model.predict(image)

        # image = cv2.imread(image)

        # table_coords = results[0].boxes
        table_coords = results[0].boxes.xyxy.numpy()  # Convert boxes to NumPy array

        x1, y1, x2, y2 = map(int, list(table_coords)[0][:4])

        # Crop table region
        table_img = image[y1:y2, x1:x2]
        return table_img

    def list_to_dict(self, table_data):
        # Extract the column names from the first row

        # Create a list of dictionaries, each representing a row
        table_dicts = []

        if len(table_data) < 2:
            return table_dicts

        first_row = table_data[0]
        for row in table_data[1:]:  # Skip the first row as it contains column names

            row_dict = {}
            for i in range(len(row)):
                row_dict[first_row[i]] = row[i]
                
            table_dicts.append(row_dict)

        return table_dicts

    def table_detection(self, img):
        # img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = cv2.bitwise_not(img_bin)

        kernel_length_v = (np.array(img_gray).shape[1])//120
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
        im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
        vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

        kernel_length_h = (np.array(img_gray).shape[1])//40
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
        im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
        table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
        thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = 0

        full_list=[]
        row=[]
        data=[]
        first_iter=0
        firsty=-1

        print ("extracting data")
        for c in tqdm(contours):
            x, y, w, h = cv2.boundingRect(c)

            # if  h > 9 and h<100:
            if first_iter==0:
                first_iter=1
                firsty=y
            if firsty!=y:
                row.reverse()
                full_list.append(row)
                row=[]
                data=[]
            # print(x,y,w,h)
            cropped = img[y:y + h, x:x + w]
            # plt.imshow(cropped)
            bounds = reader.readtext(cropped)

            try:
                data.append(bounds[0][1])
                data.append(w)
                row.append(data)
                data=[]
            except:
                data.append("")
                data.append(w)
                row.append(data)
                data=[]
            firsty=y
            # cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0), 2)
            # plt.imshow(img)
        full_list.reverse()

        new_data=[]
        new_row=[]
        for i in full_list:
            for j in i:
                new_row.append(j[0])
            new_data.append(new_row)
            new_row=[]

        print("-- new_data 1--")
        print(new_data)
        new_data = self.list_to_dict(new_data)
        print("-- new_data 2--")
        print(new_data)
        return new_data

# path = "/home/ahsan/Downloads/table.png"
# image = extract_table(path)
# data = extract_table_easyocr(image)
# print (data)