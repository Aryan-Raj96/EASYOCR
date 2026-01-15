# EasyOCR – Image to Text Extraction (Python)

## 📌 Project Overview

This project demonstrates an **end-to-end Image to Text (OCR) pipeline** using **EasyOCR**, **OpenCV**, and **Python**.
The goal is to extract readable text from images using preprocessing techniques to improve OCR accuracy.

This repository is built as a **learning + practical project**, suitable for:

* AI/ML Internship evaluation
* Computer Vision practice
* OCR experimentation

---

## 🚀 Features

* Image preprocessing (grayscale, flattening, contrast enhancement)
* Text extraction using EasyOCR
* Word cleaning and normalization
* Bounding box visualization for detected text
* Support for Indian English dictionary (`en_IN`)
* Clean and modular Python code structure

---

## 🛠️ Tech Stack

* **Python 3.x**
* **EasyOCR** – Optical Character Recognition
* **OpenCV (cv2)** – Image processing
* **NumPy** – Array operations
* **Matplotlib** – Image visualization
* **PyEnchant** – Dictionary-based text cleaning

---

## 📁 Project Structure

```
EASYOCR/
│
├── output/                 # OCR outputs
├── image.jpg               # Sample input image
├── Final_OCR_Output.txt    # Final extracted text
├── OS_OCR_output.txt       # OCR result logs
│
├── 1.py                    # Initial OCR test
├── F.py                    # OCR pipeline version
├── dic.py                  # Dictionary-based cleaning
├── dic2.py                 # Improved dictionary logic
├── down.py                 # Image download helper
├── down2.py
├── down3.py
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/EASYOCR.git
cd EASYOCR
```

### 2️⃣ Install Required Libraries

```bash
pip install easyocr opencv-python matplotlib numpy pyenchant
```

⚠️ **Note (Windows users):**
If `pyenchant` throws an error, install the enchant backend separately.

---

## ▶️ How to Run the Project

1. Place your input image in the project folder (example: `image.jpg`)
2. Open the Python file containing the OCR pipeline
3. Run the script:

```bash
python your_script_name.py
```

4. The program will:

   * Show intermediate preprocessing steps
   * Display OCR bounding boxes
   * Print extracted text in the terminal

---

## 🧠 OCR Pipeline Explained

1. **Image Loading**
2. **Grayscale Conversion**
3. **Flattening (Illumination normalization)**
4. **Contrast Enhancement (CLAHE)**
5. **Text Detection using EasyOCR**
6. **Text Cleaning & Normalization**
7. **Bounding Box Visualization**

---

## 📊 Sample Output

* **Raw OCR Words** – Direct EasyOCR output
* **Cleaned Words** – Alphabet-only, normalized words
* **Bounding Boxes** – Visual representation of detected text

---

## 💡 Learning Outcomes

* Practical understanding of OCR systems
* Importance of preprocessing in computer vision
* Integrating ML models into real-world pipelines
* Writing clean, readable, and modular Python code

---

## 🔮 Future Improvements

* Web interface using Flask
* Image upload from frontend
* Confidence-score based filtering
* Deployment on cloud (Render / Railway)
* Multi-language OCR support

---

## 👤 Author

**Aryan Raj**
AI/ML Enthusiast | Computer Vision Learner

---

## Acknowledgements

* EasyOCR open-source community
* OpenCV documentation
* Python ecosystem


## License

This project is for **educational and learning purposes**.

✅ *If you find this project useful, feel free to star the repository!* ⭐
