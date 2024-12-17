import cv2
import pytesseract
from ultralytics import YOLO
from ultralyticsplus import YOLO, render_result
import easyocr
import numpy as np

class ocr_table:

    def __init__(self):
        pass

    def extract_table(self, image_path):

        # load model
        model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        model.overrides['conf'] = 0.25  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000  # maximum number of detections per image

        # perform inference
        results = model.predict(image_path)

        image = cv2.imread(image_path)

        # table_coords = results[0].boxes
        table_coords = results[0].boxes.xyxy.numpy()  # Convert boxes to NumPy array

        x1, y1, x2, y2 = map(int, list(table_coords)[0][:4])

        # Crop table region
        table_img = image[y1:y2, x1:x2]
        return table_img


    def rows_count(self, image):
        # Initialize EasyOCR Reader
        reader = easyocr.Reader(['en'])  # English language
        
        # Load the image
        # image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not read image. Check the file path.")
            return []

        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize the image for better OCR accuracy
        scale_percent = 200  # Increase size by 200%
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized_gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Apply thresholding
        _, thresh = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find horizontal lines (table rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # Sort rows top-to-bottom

        return len(contours) - 1

    def list_to_dict(self, table_data):
        # Extract the column names from the first row
        column_names = table_data[0]

        # Create a list of dictionaries, each representing a row
        table_dicts = []
        for row in table_data[1:]:  # Skip the first row as it contains column names
            row_dict = {column_names[i]: row[i] for i in range(len(column_names))}
            table_dicts.append(row_dict)

        return table_dicts
        
    def extract_table_easyocr(self, image):

        total_rows = self.rows_count(image)    
        reader = easyocr.Reader(['en'])  # English language
        
        # Load the image
        # image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image is None:
            print("Error: Could not read image. Check the file path.")
            return []
        
        # Resize the image for better OCR accuracy
        scale_percent = 200  # Increase size by 200%
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized_gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Apply thresholding
        _, thresh = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Debugging step: Save intermediate processed image
        cv2.imwrite("processed_image.png", thresh)
        # print("Processed image saved for verification: 'processed_image.png'")

        # Find contours to detect rows
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # Sort top-to-bottom

        print(f"Number of detected contours (rows): {len(contours)}")
        if not contours:
            print("Error: No contours (rows) detected in the image.")
            return []

        rows_data = []  # List to store row-wise text

        # Loop through each detected row and perform OCR
        for i, ctr in enumerate(contours):
            x, y, w, h = cv2.boundingRect(ctr)

            row_image = resized_gray[y:y+h, x:x+w]  # Crop the row
            # Perform OCR on the row
            result = reader.readtext(row_image, detail=0)  # Only text, no bounding boxes
            # print(f"OCR result for row {i}: {result}")

            if result:  # Append non-empty rows
                rows_data.append(result)

        rows = total_rows
        columns = int(len(rows_data[0])/total_rows)

        print ("-- rows_data --")
        print (rows_data)
        print ("-- rows_data --")

        data = np.array(rows_data[0])
        data = data.reshape(rows, columns)
        data = self.list_to_dict(data)
        
        return data

# path = "/home/ahsan/Downloads/table.png"
# image = extract_table(path)
# data = extract_table_easyocr(image)
# print (data)