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

class EMeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnaMenu()
        self.ui.setupUi(self)
        self.ui.pushButton_WhiteListEdit.clicked.connect(self.openwlfile)
        self.ui.pushButton_KameraAc.clicked.connect(self.opencam)
        self.ui.pushButton_MedyaSec.clicked.connect(self.openmedia)
        self.ui.pushButton_Basla.clicked.connect(self.startprocessing)
        self.ui.pushButton_Model_Egit.clicked.connect(self.train_whitelist_model)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.is_camera = False
        self.is_processing = False
        self.current_media_frame = None
        self.original_media_frame = None
        
        # Optimizasyon değişkenleri
        self.last_face_detection_time = 0
        self.face_detection_interval = 0.3  # Saniyede 2 kez yüz tespiti
        self.cached_faces = []
        self.processed_frame_count = 0
        self.frame_skip_ratio = 1  # 2 frame'de 1 işlem yap
        
        self.whitelist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiteList")
        if not os.path.exists(self.whitelist_dir):
            os.makedirs(self.whitelist_dir)
        
        self.whitelist_embeddings = []
        
        # YOLO modelini yükle
        self.yolo_model = None
        self.load_yolo_model()
        
        # CUDA desteğini kontrol et
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def load_yolo_model(self):
        """YOLO yüz tespit modelini yükle"""
        try:
            # YOLOv8 face detection modeli
            self.yolo_model = YOLO('yolov8n-face.pt')  # Nano model - daha hızlı
            # Alternatif modeller: yolov8s-face.pt, yolov8m-face.pt (daha doğru ama daha yavaş)
            print("YOLO model başarıyla yüklendi")
        except Exception as e:
            print(f"YOLO model yükleme hatası: {e}")
            QMessageBox.warning(self, "Hata", "YOLO modeli yüklenemedi!")
            self.yolo_model = None

    def closeEvent(self, event):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        event.accept()

    def openmedia(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Medya Seç", "",
                                                   "Media Files (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)")
        if not file_path:
            return
        self.is_camera = False
        self.ui.pushButton_KameraAc.setText("Kamerayı Aç")
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)
            self.ui.pushButton_Basla.setEnabled(True)
        else:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.timer.stop()
            frame = cv2.imread(file_path)
            if frame is None:
                return
            frame = self.resize_frame(frame, max_width=800)
            self.current_media_frame = frame.copy()
            self.original_media_frame = frame.copy()
            self.display_frame(frame)
            self.ui.pushButton_Basla.setEnabled(True)

    def resize_frame(self, frame, max_width=800):
        """Görüntü boyutunu optimize et"""
        height, width = frame.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def opencam(self):
        if self.cap is None:
            self.is_camera = True
            self.ui.pushButton_Basla.setEnabled(True)
            self.cap = cv2.VideoCapture(0)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "Hata", "Kamera açılamadı!")
                self.cap = None
                return
            self.timer.start(30)
            self.ui.pushButton_KameraAc.setText("Kamerayı Kapat")
        else:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.ui.Ekran.clear()
            self.ui.pushButton_Basla.setEnabled(False)
            self.ui.pushButton_KameraAc.setText("Kamerayı Aç")
            self.is_processing = False
            self.ui.pushButton_Basla.setText("Başlat")

    def update_frame(self):
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        if self.is_camera:
            frame = cv2.flip(frame, 1)
        
        self.processed_frame_count += 1
        if self.processed_frame_count % self.frame_skip_ratio != 0 and self.is_processing:
            self.display_frame(frame)
            return
        
        if self.is_processing:
            processed_frame = self.yolo_detect_faces(frame)
            self.display_frame(processed_frame)
        else:
            self.display_frame(frame)

    def yolo_detect_faces(self, frame):
        """YOLO ile yüz tespiti ve tanıma"""
        if self.yolo_model is None:
            return frame
        
        frame_copy = frame.copy()
        
        # YOLO ile yüz tespiti
        results = self.yolo_model(frame_copy, conf=0.5, verbose=False, device=self.device)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Bounding box koordinatları
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Yüz bölgesini al
                    face_region = frame_copy[y1:y2, x1:x2]
                    
                    if face_region.size == 0:
                        continue
                    
                    # Yüzü tanıma ve renk belirleme
                    color, similarity = self.recognize_face(face_region)
                    
                    # Bounding box çiz
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiket ekle
                    label = f"{'Known' if color == (0, 255, 0) else 'Unknown'} {similarity:.2f}"
                    cv2.putText(frame_copy, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy

    def recognize_face(self, face_region):
        """Yüzü tanıma ve whitelist ile karşılaştırma"""
        color = (0, 0, 255)  # Kırmızı: Tanınmadı
        similarity = 0.0
        
        if not self.whitelist_embeddings:
            return color, similarity
        
        try:
            # Embedding çıkar
            embedding_result = DeepFace.represent(
                img_path=face_region,
                model_name='Facenet',
                detector_backend='skip',
                enforce_detection=False,
                align=True
            )
            
            if embedding_result:
                face_embedding = np.array(embedding_result[0]['embedding'])
                
                # Whitelist'teki embedding'lerle karşılaştır
                max_similarity = 0.0
                for w_emb in self.whitelist_embeddings:
                    w_embedding = np.array(w_emb['embedding'])
                    current_similarity = self.cosine_similarity_fast(face_embedding, w_embedding)
                    max_similarity = max(max_similarity, current_similarity)
                    
                    if current_similarity > 0.75:  # Eşik değeri
                        color = (0, 255, 0)  # Yeşil: Tanındı
                        break
                
                similarity = max_similarity
                
        except Exception as e:
            print(f"Yüz tanıma hatası: {e}")
        
        return color, similarity

    def cosine_similarity_fast(self, a, b):
        """Hızlı cosine similarity hesaplama"""
        a = a.flatten()
        b = b.flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    def train_whitelist_model(self):
        """Whitelist modelini eğit"""
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
                    
                    # YOLO ile yüz tespiti (daha doğru tespit için)
                    if self.yolo_model:
                        results = self.yolo_model(img, conf=0.5, verbose=False, device=self.device)
                        for result in results:
                            if result.boxes is not None and len(result.boxes) > 0:
                                box = result.boxes[0]  # İlk yüzü al
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                face_region = img[y1:y2, x1:x2]
                                
                                if face_region.size == 0:
                                    continue
                                
                                # Embedding çıkar
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
                        # Fallback: DeepFace ile yüz tespiti
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

    def startprocessing(self):
        if not self.whitelist_embeddings:
            QMessageBox.warning(self, "Uyarı", "Önce modeli eğitin!")
            return
            
        self.is_processing = not self.is_processing
        if self.is_processing:
            self.ui.pushButton_Basla.setText("Durdur")
            if not self.is_camera and self.current_media_frame is not None and self.cap is None:
                processed_frame = self.yolo_detect_faces(self.current_media_frame.copy())
                self.display_frame(processed_frame)
        else:
            self.ui.pushButton_Basla.setText("Başlat")
            if not self.is_camera and self.original_media_frame is not None and self.cap is None:
                self.display_frame(self.original_media_frame)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EMeApp()
    window.show()
    sys.exit(app.exec_())