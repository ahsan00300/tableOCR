import os
from flask import request, jsonify, Blueprint

from app.services.OCR.table_extraction_pipeline import ocr_table


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr_blueprint = Blueprint('upload', __name__)

@ocr_blueprint.route('/upload', methods=['POST'])
def upload_file():

    return_list = []
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process file for table extraction
    try:

        # path = "/home/ahsan/Downloads/table.png"
        ocr_obj = ocr_table()

        pdf_pages = ocr_obj.pdf_to_images(filepath)

        for pdf_page in pdf_pages:
            image = ocr_obj.extract_table(pdf_page)
            if len(image) > 0:
                table_data = ocr_obj.table_detection(image)
                return_list = table_data
                break
        
        print ("-- return_list --")
        print (return_list)
        return return_list
    except Exception as e:
        return jsonify({"error": str(e)}), 500
