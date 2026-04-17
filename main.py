import sys
from PySide6.QtWidgets import QApplication
from view import MainWindow
from model import MeshModel, LensModel
from controller import Controller

def main():
    app = QApplication(sys.argv)
    
    view = MainWindow()
    mesh = MeshModel(rows=5, cols=5)
    lens = LensModel()
    
    controller = Controller(view, mesh, lens)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
