from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from EMeUi import Ui_AnaMenu
import os
import cv2
import numpy as np
import pickle
from PyQt5.QtWidgets import QFileDialog
import face_recognition
from sklearn.preprocessing import LabelEncoder

class EMeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnaMenu()
        self.ui.setupUi(self)
        self.ui.pushButton_WhiteListEdit.clicked.connect(self.openwlfile)
        self.ui.pushButton_Kaydedilenler.clicked.connect(self.opensavefile)
        self.ui.pushButton_KameraAc.clicked.connect(self.opencam)
        self.ui.pushButton_MedyaSec.clicked.connect(self.openmedia)
        self.ui.pushButton_Basla.clicked.connect(self.startprocessing)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.is_camera = False
        self.is_processing = False
        self.current_media_frame = None
        self.original_media_frame = None
        
        # Yüz tanıma için yeni değişkenler
        self.known_face_encodings = []
        self.known_face_names = []
        self.label_encoder = LabelEncoder()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_whitelist_faces()

        face_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FaceR")
        deploy_path = os.path.join(face_model_dir, "deploy.prototxt")
        model_path = os.path.join(face_model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(deploy_path) or not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "Hata", "FaceR klasöründe model dosyaları eksik!")
            sys.exit()

        self.face_net = cv2.dnn.readNetFromCaffe(deploy_path, model_path)

    def load_whitelist_faces(self):
        """WhiteList klasöründeki yüzleri yükler - Geliştirilmiş"""
        program_dir = os.path.dirname(os.path.abspath(__file__))
        whitelist_dir = os.path.join(program_dir, "WhiteList")
        
        if not os.path.exists(whitelist_dir):
            os.makedirs(whitelist_dir)
            return
        
        labels_file = os.path.join(whitelist_dir, "whitelist_data.pkl")
        
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"WhiteList'ten {len(self.known_face_encodings)} yüz yüklendi")
            except Exception as e:
                print(f"Model yüklenirken hata: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            self.train_whitelist_model(whitelist_dir)
    
    def extract_face_features(self, face_image):
        """Yüz özellikleri çıkar - Geliştirilmiş (face_recognition)"""
        if face_image.size == 0:
            return None
        
        try:
            # Gri görüntüyü RGB'ye çevir
            if len(face_image.shape) == 2:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Yüz embedding'leri çıkar
            encodings = face_recognition.face_encodings(rgb_face)
            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"Yüz encoding hatası: {e}")
        
        return None

    def compare_faces(self, face_features1, face_features2):
        """İki yüzü karşılaştır - Cosine similarity"""
        if face_features1 is None or face_features2 is None:
            return 0.0
        
        # Cosine benzerliği
        dot_product = np.dot(face_features1, face_features2)
        norm1 = np.linalg.norm(face_features1)
        norm2 = np.linalg.norm(face_features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def train_whitelist_model(self, whitelist_dir):
        """WhiteList klasöründeki resimlerden model eğit - Geliştirilmiş"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for item_name in os.listdir(whitelist_dir):
            item_path = os.path.join(whitelist_dir, item_name)
            
            if os.path.isdir(item_path):
                # Klasör içindeki resimleri işle
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(item_path, filename)
                        self.process_training_image(image_path, item_name)
            elif item_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Doğrudan resim dosyalarını işle
                person_name = os.path.splitext(item_name)[0]
                self.process_training_image(item_path, person_name)
        
        self.save_training_data(whitelist_dir)

    def process_training_image(self, image_path, person_name):
        """Eğitim görüntüsünü işle"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return
            
            # Yüzleri bul ve encode et
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            for encoding in face_encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)
                
        except Exception as e:
            print(f"{image_path} işlenirken hata: {e}")

    def save_training_data(self, whitelist_dir):
        """Eğitim verilerini kaydet"""
        labels_file = os.path.join(whitelist_dir, "whitelist_data.pkl")
        
        with open(labels_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        
        print(f"Model eğitildi: {len(self.known_face_encodings)} yüz örneği")

    def recognize_face(self, face_image):
        """Yüzü tanı - Geliştirilmiş"""
        if not self.known_face_encodings:
            return None, 0.0
        
        # Giriş yüzünü encode et
        input_encoding = self.extract_face_features(face_image)
        if input_encoding is None:
            return None, 0.0
        
        # Tüm bilinen yüzlerle karşılaştır
        best_similarity = 0.0
        best_name = None
        
        for i, known_encoding in enumerate(self.known_face_encodings):
            similarity = self.compare_faces(input_encoding, known_encoding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = self.known_face_names[i]
        
        # Eşik değeri
        if best_similarity > 0.6:  # %60 benzerlik eşiği
            return best_name, best_similarity
        
        return None, best_similarity

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
            self.ui.pushButton_Kaydet.setEnabled(True)
        else:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.timer.stop()
            frame = cv2.imread(file_path)
            if frame is None:
                return
            self.current_media_frame = frame.copy()
            self.original_media_frame = frame.copy()
            self.display_frame(frame)
            self.ui.pushButton_Basla.setEnabled(True)

    def opencam(self):
        if self.cap is None:
            self.is_camera = True
            self.ui.pushButton_Basla.setEnabled(True)
            self.cap = cv2.VideoCapture(0)
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
            self.ui.pushButton_Kaydet.setEnabled(False)
            self.ui.pushButton_KameraAc.setText("Kamerayı Aç")
            self.is_processing = False
            self.ui.pushButton_Basla.setText("Başlat")

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            if self.is_camera:
                frame = cv2.flip(frame, 1)
            
            if self.is_processing:
                processed_frame = self.detect_faces(frame)
                self.display_frame(processed_frame)
            else:
                self.display_frame(frame)

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        if self.known_face_encodings:
            return self.detect_and_compare_faces(frame, detections, h, w)
        else:
            return self.detect_faces_normal(frame, detections, h, w)
    
    def detect_and_compare_faces(self, frame, detections, h, w):
        """Yüzleri algıla ve WhiteList ile karşılaştır - Geliştirilmiş"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.20:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                
                # Yüz bölgesini biraz küçült
                box_width = endX - startX
                box_height = endY - startY
                new_width = int(box_width * 0.80)
                new_height = int(box_height * 0.80)
                center_x = startX + box_width // 2
                center_y = startY + box_height // 2
                startX = max(0, center_x - new_width // 2)
                startY = max(0, center_y - new_height // 2)
                endX = min(w, center_x + new_width // 2)
                endY = min(h, center_y + new_height // 2)
                
                face_roi = gray_frame[startY:endY, startX:endX]
                if face_roi.size > 0:
                    try:
                        name, similarity = self.recognize_face(face_roi)
                        similarity_percent = int(similarity * 100)
                        
                        if name is not None:
                            if similarity_percent > 75:  # Yüksek güvenilirlik
                                color = (0, 255, 0)
                                label_text = f"WL: {name} ({similarity_percent}%)"
                            else:  # Orta güvenilirlik
                                color = (255, 165, 0)  # Turuncu
                                label_text = f"Şüpheli: {name} ({similarity_percent}%)"
                        else:
                            color = (0, 0, 255)
                            label_text = f"Bilinmeyen ({similarity_percent}%)"
                    except Exception as e:
                        print(f"Tanıma hatası: {e}")
                        color = (0, 0, 255)  
                        label_text = "Hata"
                else:
                    color = (0, 0, 255)
                    label_text = "Görüntü yok"
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label_text, (startX, startY - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def detect_faces_normal(self, frame, detections, h, w):
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.20:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return frame

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
        program_dir = os.path.dirname(os.path.abspath(__file__))
        wlfolder = os.path.join(program_dir, "WhiteList")
        if not os.path.exists(wlfolder):
            os.makedirs(wlfolder)
        os.startfile(wlfolder)
        
        QtCore.QTimer.singleShot(2000, self.refresh_whitelist)

    def refresh_whitelist(self):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        whitelist_dir = os.path.join(program_dir, "WhiteList")
        
        labels_file = os.path.join(whitelist_dir, "whitelist_data.pkl")
        
        if os.path.exists(labels_file):
            os.remove(labels_file)
        
        self.train_whitelist_model(whitelist_dir)
        QtWidgets.QMessageBox.information(self, "Bilgi", "WhiteList yenilendi!")

    def opensavefile(self):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        savefolder = os.path.join(program_dir, "Saved")
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        os.startfile(savefolder)

    def startprocessing(self):
        self.is_processing = not self.is_processing
        
        if self.is_processing:
            self.ui.pushButton_Kaydet.setEnabled(True)
            self.ui.pushButton_Basla.setText("Durdur")
            if not self.is_camera and self.current_media_frame is not None and self.cap is None:
                processed_frame = self.detect_faces(self.current_media_frame.copy())
                self.display_frame(processed_frame)
        else:
            self.ui.pushButton_Basla.setText("Başlat")
            self.ui.pushButton_Kaydet.setEnabled(False)
            if not self.is_camera and self.original_media_frame is not None and self.cap is None:
                self.display_frame(self.original_media_frame)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EMeApp()
    window.show()
    sys.exit(app.exec_())