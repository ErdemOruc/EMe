from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from EMeUi import Ui_AnaMenu
import os
import cv2
from PyQt5.QtWidgets import QFileDialog

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

        face_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FaceR")
        deploy_path = os.path.join(face_model_dir, "deploy.prototxt")
        model_path = os.path.join(face_model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(deploy_path) or not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "Hata", "FaceR klasöründe model dosyaları eksik!")
            sys.exit()

        self.face_net = cv2.dnn.readNetFromCaffe(deploy_path, model_path)

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
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.20:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                box_width = endX - startX
                box_height = endY - startY
                new_width = int(box_width * 0.70)
                new_height = int(box_height * 0.70)
                center_x = startX + box_width // 2
                center_y = startY + box_height // 2
                startX = center_x - new_width // 2
                startY = center_y - new_height // 2
                endX = center_x + new_width // 2
                endY = center_y + new_height // 2
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
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