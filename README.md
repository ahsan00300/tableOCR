# Table Extraction with OCR

This project provides a Flask API to extract tables from images using OCR (Optical Character Recognition). It allows users to upload a pdf containing tables, processes the pdf, extract tables from pdf and returns the extracted table data in JSON format.

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Running the Flask Server](#running-the-flask-server)  
5. [API Usage](#api-usage)  
6. [Folder Structure](#folder-structure)  
7. [License](#license)  

---

## Features

- Upload a pdf file containing tables.
- Extract tables using OCR.
- Return the extracted table data as JSON.

---

## Requirements

- Python 3.8 or higher  
- YOLO (for table detection)
- Flask  
- EasyOCR  
- OpenCV (Optional but recommended for image processing tasks)  

---

## Installation

Follow the steps below to set up the project on your local machine:

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd <project_directory>

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows

   pip install -r requirements.txt

   python main.py


## Endpoint

http://127.0.0.1:8002/upload

- endpoint type is "POST"
- endpoint payload is form data file
- endpint payload key is keyword 'file'
- endpoint input is pdf file


## Output

list of dictionaries where each dictionary contains values of one row

### sample outout

[{"height":"15","width":"20","length":"10"},
{"height":"11","width":"21","length":"15"}]
