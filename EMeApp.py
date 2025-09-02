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
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.is_camera = False

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qt_img)
            scaled_pix = pix.scaled(self.ui.Ekran.width(), self.ui.Ekran.height(),
                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.ui.Ekran.setPixmap(scaled_pix)
            self.ui.Ekran.setAlignment(QtCore.Qt.AlignCenter)
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
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            if self.is_camera:
                frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            if not qt_img.isNull():
                pix = QtGui.QPixmap.fromImage(qt_img)
                scaled_pix = pix.scaled(self.ui.Ekran.width(), self.ui.Ekran.height(),
                                      QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.ui.Ekran.setPixmap(scaled_pix)
                self.ui.Ekran.setAlignment(QtCore.Qt.AlignCenter)

    def openwlfile(self):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        wlfolder = os.path.join(program_dir, "WhiteList")
        if os.path.exists(wlfolder):
            os.startfile(wlfolder)
        else:
            os.makedirs(wlfolder)
            os.startfile(wlfolder)

    def opensavefile(self):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        wlfolder = os.path.join(program_dir, "Saved")
        if os.path.exists(wlfolder):
            os.startfile(wlfolder)
        else:
            os.makedirs(wlfolder)
            os.startfile(wlfolder)

app = QtWidgets.QApplication(sys.argv)
window = EMeApp()
window.show()
sys.exit(app.exec_())