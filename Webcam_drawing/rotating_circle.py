import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont


class RotatingCircle(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generating Image")
        self.resize(1000, 1000)
        self.angle = 0

        # Timer to trigger rotation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(16)  # ~60 FPS

    def update_angle(self):
        self.angle = (self.angle + 2) % 360  # Rotate by 2 degrees per frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        radius = 400
        # Translate to center, rotate, then draw
        cx, cy = self.width() // 2, self.height() // 2


         # Move this block before painter.rotate(self.angle)
        painter.resetTransform()  # Optional, or create a second QPainter
        painter.setPen(QPen(Qt.black))
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        text = "Génération d'images en cours ... "
        text_width = painter.fontMetrics().width(text)
        #print(cy, radius)
        painter.drawText(cx - text_width // 2, 80, text)

        painter.translate(cx, cy)
        painter.rotate(self.angle)

        # Draw the rotating circle (static)
        
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(QColor("lightblue")))
        painter.drawEllipse(-radius, -radius, radius * 2, radius * 2)

        # Draw a line to show rotation
        painter.setPen(QPen(Qt.red, 4))
        painter.drawLine(0, 0, radius, 0)  # Rotates with the circle



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RotatingCircle()
    window.show()
    sys.exit(app.exec_())
