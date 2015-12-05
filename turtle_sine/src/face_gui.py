#!/usr/bin/env python
import cv2
from PyQt4 import QtGui, QtCore


class Capture():
    def __init__(self):
        self.capturing = False
        self.c = cv2.VideoCapture(0)

    def startCapture(self):
        print "Camera turned on!"
        self.capturing = True
        cap = self.c
        while(self.capturing):
            ret, frame = cap.read()
            self.ds_factor = 0.5
            frame = cv2.resize(frame, None, fx=self.ds_factor, fy=self.ds_factor, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Capture", frame)
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def endCapture(self):
        print "Camera turned off!"
        self.capturing = False

    def quitCapture(self):
        print "Pressed Quit!"
        cap = self.c
        cv2.destroyAllWindows()
        cap.release()
        QtCore.QCoreApplication.quit()


class Window(QtGui.QWidget):
    def __init__(self):

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.capture = Capture()
        self.start_button = QtGui.QPushButton('Start Camera',self)
        self.start_button.clicked.connect(self.capture.startCapture)

        self.end_button = QtGui.QPushButton('End Camera',self)
        self.end_button.clicked.connect(self.capture.endCapture)

        self.quit_button = QtGui.QPushButton('Quit Camera',self)
        self.quit_button.clicked.connect(self.capture.quitCapture)

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        self.setLayout(vbox)
        self.setGeometry(100,100,200,200)
        self.show()


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())