from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from EMeUi import Ui_AnaMenu
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from deepface import DeepFace
from ultralytics import YOLO
import torch
import pickle
import hashlib

class EMeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnaMenu()
        self.ui.setupUi(self)
        self.ui.pushButton_WhiteListEdit.clicked.connect(self.openwlfile)
        self.ui.pushButton_MedyaSec.clicked.connect(self.openmedia)
        self.ui.pushButton_Basla.clicked.connect(self.startprocessing)
        self.ui.pushButton_Model_Egit.clicked.connect(self.train_whitelist_model)
        self.is_processing = False
        self.current_media_frame = None
        self.original_media_frame = None
        self.whitelist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiteList")
        if not os.path.exists(self.whitelist_dir):
            os.makedirs(self.whitelist_dir)
        self.whitelist_embeddings = []
        self.embeddings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelist_embeddings.pkl")
        self.whitelist_hash_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelist_hash.txt")
        self.yolo_model = None
        self.load_yolo_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.load_embeddings_if_valid()

    def calculate_whitelist_hash(self):
        image_files = [f for f in os.listdir(self.whitelist_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return "empty"
        file_info = []
        for file in sorted(image_files):
            file_path = os.path.join(self.whitelist_dir, file)
            file_size = os.path.getsize(file_path)
            file_info.append(f"{file}_{file_size}")
        
        return hashlib.md5("_".join(file_info).encode()).hexdigest()

    def load_embeddings_if_valid(self):
        current_hash = self.calculate_whitelist_hash()
        
        try:
            with open(self.whitelist_hash_file, 'r') as f:
                saved_hash = f.read().strip()
            
            if current_hash == saved_hash and os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.whitelist_embeddings = pickle.load(f)
                print("Önceden eğitilmiş embedding'ler yüklendi")
                QMessageBox.information(self, "Bilgi", "Önceden eğitilmiş model yüklendi!")
                return True
                
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
        
        return False

    def save_embeddings_and_hash(self):
        current_hash = self.calculate_whitelist_hash()
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.whitelist_embeddings, f)
        
        with open(self.whitelist_hash_file, 'w') as f:
            f.write(current_hash)
        
        print("Embedding'ler ve hash kaydedildi")

    def load_yolo_model(self):
        try:
            self.yolo_model = YOLO('yolov8n-face.pt') # yolov8n-face.pt / yolov8m-face.pt / yolov8l-face.pt
            
        
            print("YOLO model başarıyla yüklendi")
        except Exception as e:
            print(f"YOLO model yükleme hatası: {e}")
            QMessageBox.warning(self, "Hata", "YOLO modeli yüklenemedi!")
            self.yolo_model = None

    def openmedia(self):
        if self.is_processing:
            self.stop_processing()
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.JPG *.JPEG)")
        if not file_path:
            return
        
        frame = cv2.imread(file_path)
        if frame is None:
            QMessageBox.warning(self, "Hata", "Resim yüklenemedi!")
            return
            
        frame = self.resize_frame(frame, max_width=800)
        self.current_media_frame = frame.copy()
        self.original_media_frame = frame.copy()
        self.display_frame(frame)
        self.ui.pushButton_Basla.setEnabled(True)
        self.ui.pushButton_Basla.setText("Başlat")
        self.is_processing = False

    def resize_frame(self, frame, max_width=800):
        height, width = frame.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def yolo_detect_faces(self, frame):
        if self.yolo_model is None:
            return frame
        
        frame_copy = frame.copy()
        
        results = self.yolo_model(frame_copy, conf=0.5, verbose=False, device=self.device)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    face_region = frame_copy[y1:y2, x1:x2]
                    
                    if face_region.size == 0:
                        continue
                    
                    color, similarity = self.recognize_face(face_region)
                    
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{'Known' if color == (0, 255, 0) else 'Unknown'} {similarity:.2f}"
                    cv2.putText(frame_copy, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy

    def recognize_face(self, face_region):
        color = (0, 0, 255)
        similarity = 0.0
        
        if not self.whitelist_embeddings:
            return color, similarity
        
        try:
            embedding_result = DeepFace.represent(
                img_path=face_region,
                model_name='Facenet',
                detector_backend='skip',
                enforce_detection=False,
                align=True
            )
            
            if embedding_result:
                face_embedding = np.array(embedding_result[0]['embedding'])
                
                max_similarity = 0.0
                for w_emb in self.whitelist_embeddings:
                    w_embedding = np.array(w_emb['embedding'])
                    current_similarity = self.cosine_similarity_fast(face_embedding, w_embedding)
                    max_similarity = max(max_similarity, current_similarity)
                    
                    if current_similarity > 0.70: #Eşik değeri
                        color = (0, 255, 0)
                        break
                
                similarity = max_similarity
                
        except Exception as e:
            print(f"Yüz tanıma hatası: {e}")
        
        return color, similarity

    def cosine_similarity_fast(self, a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    def train_whitelist_model(self):
        if self.load_embeddings_if_valid():
            if QMessageBox.question(self, "Soru", 
                                  "Model zaten güncel görünüyor. Yine de yeniden eğitmek istiyor musunuz?",
                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        
        try:
            self.whitelist_embeddings = []
            image_files = [f for f in os.listdir(self.whitelist_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                QMessageBox.warning(self, "Uyarı", "WhiteList klasöründe resim bulunamadı!")
                return
            
            progress_dialog = QtWidgets.QProgressDialog("Yüzler eğitiliyor...", "İptal", 0, len(image_files), self)
            progress_dialog.setWindowTitle("Model Eğitimi")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            progress_dialog.show()
            
            successful_count = 0
            
            for i, file in enumerate(image_files):
                if progress_dialog.wasCanceled():
                    break
                    
                img_path = os.path.join(self.whitelist_dir, file)
                progress_dialog.setLabelText(f"İşleniyor: {file}")
                progress_dialog.setValue(i)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = self.resize_frame(img, max_width=400)
                    if self.yolo_model:
                        results = self.yolo_model(img, conf=0.5, verbose=False, device=self.device)
                        for result in results:
                            if result.boxes is not None and len(result.boxes) > 0:
                                box = result.boxes[0]
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                face_region = img[y1:y2, x1:x2]
                                
                                if face_region.size == 0:
                                    continue
                                
                                embedding_result = DeepFace.represent(
                                    img_path=face_region,
                                    model_name='Facenet',
                                    detector_backend='skip',
                                    enforce_detection=False,
                                    align=True
                                )
                                
                                if embedding_result:
                                    self.whitelist_embeddings.append({
                                        'path': img_path,
                                        'embedding': embedding_result[0]['embedding'],
                                        'filename': file
                                    })
                                    successful_count += 1
                                    print(f"Başarılı: {file}")
                                    break
                    else:
                        embedding_result = DeepFace.represent(
                            img_path=img,
                            model_name='Facenet',
                            detector_backend='mtcnn',
                            enforce_detection=True,
                            align=True
                        )
                        
                        if embedding_result:
                            self.whitelist_embeddings.append({
                                'path': img_path,
                                'embedding': embedding_result[0]['embedding'],
                                'filename': file
                            })
                            successful_count += 1
                            print(f"Başarılı: {file}")
                
                except Exception as e:
                    print(f"Hata: {file} - {str(e)}")
                    continue
            
            progress_dialog.close()
            
            if successful_count > 0:
                # Embedding'leri ve hash'i kaydet
                self.save_embeddings_and_hash()
                QMessageBox.information(self, "Model Eğitimi", 
                                      f"{successful_count}/{len(image_files)} yüz başarıyla eğitildi!")
                print(f"Toplam {len(self.whitelist_embeddings)} embedding kaydedildi")
            else:
                QMessageBox.warning(self, "Uyarı", "Hiçbir yüz eğitilemedi!")
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model eğitimi sırasında hata: {str(e)}")
            print(f"Eğitim hatası: {e}")

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_img)
        scaled_pix = pix.scaled(self.ui.Ekran.width(), self.ui.Ekran.height(),
                                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.ui.Ekran.setPixmap(scaled_pix)
        self.ui.Ekran.setAlignment(QtCore.Qt.AlignCenter)

    def openwlfile(self):
        if not os.path.exists(self.whitelist_dir):
            os.makedirs(self.whitelist_dir)
        os.startfile(self.whitelist_dir)

    def stop_processing(self):
        self.is_processing = False
        self.ui.pushButton_Basla.setText("Başlat")
        if self.original_media_frame is not None:
            self.display_frame(self.original_media_frame)

    def startprocessing(self):
        if not self.whitelist_embeddings:
            QMessageBox.warning(self, "Uyarı", "Önce modeli eğitin!")
            return
            
        if self.current_media_frame is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir resim seçin!")
            return
            
        self.is_processing = not self.is_processing
        
        if self.is_processing:
            self.ui.pushButton_Basla.setText("Durdur")
            processed_frame = self.yolo_detect_faces(self.current_media_frame.copy())
            self.display_frame(processed_frame)
        else:
            self.stop_processing()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EMeApp()
    window.show()
    sys.exit(app.exec_())