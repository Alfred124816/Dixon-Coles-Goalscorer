import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """Initialize and run the Dixon-Coles prediction application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 