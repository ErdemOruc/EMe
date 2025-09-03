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
        self.current_frame = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
            self.current_frame = frame.copy()
            self.show_frame(frame)
            self.ui.pushButton_Basla.setEnabled(True)
            self.ui.pushButton_Kaydet.setEnabled(True)

    def opencam(self):
        if self.cap is None:
            self.is_camera = True
            self.ui.pushButton_Basla.setEnabled(True)
            self.ui.pushButton_Kaydet.setEnabled(True)
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

    def update_frame(self):
        if self.cap is None and self.current_frame is None:
            return

        if self.is_camera:
            ret, frame = self.cap.read()
            if not ret:
                return
            frame = cv2.flip(frame, 1)
            if self.is_processing:
                frame = self.detect_faces(frame)
            self.show_frame(frame)
        else:
            frame = self.current_frame.copy()
            if self.is_processing:
                frame = self.detect_faces(frame)
            self.show_frame(frame)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
        for (x, y, w, h) in faces:
            if w < 30 or h < 30:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_img)
        scaled_pix = pix.scaled(self.ui.Ekran.width(), self.ui.Ekran.height(),
                                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.ui.Ekran.setPixmap(scaled_pix)
        self.ui.Ekran.setAlignment(QtCore.Qt.AlignCenter)

    def startprocessing(self):
        self.is_processing = not self.is_processing
        self.ui.pushButton_Basla.setText("Durdur" if self.is_processing else "Başlat")
        if not self.is_camera and self.current_frame is not None:
            if self.is_processing:
                frame = self.detect_faces(self.current_frame.copy())
            else:
                frame = self.current_frame.copy()
            self.show_frame(frame)

    def openwlfile(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiteList")
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.startfile(folder)

    def opensavefile(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saved")
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.startfile(folder)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EMeApp()
    window.show()
    sys.exit(app.exec_())
