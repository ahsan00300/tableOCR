import os
from flask import request, jsonify, Blueprint

from app.services.OCR.table_extraction_pipeline import ocr_table


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr_blueprint = Blueprint('upload', __name__)

@ocr_blueprint.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process file for table extraction
    try:

        # path = "/home/ahsan/Downloads/table.png"
        ocr_obj = ocr_table()
        image = ocr_obj.extract_table(filepath)
        table_data = ocr_obj.extract_table_easyocr(image)
        
        return jsonify(table_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
