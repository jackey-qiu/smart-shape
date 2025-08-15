import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication
from ..gui.main_gui import smartShapeGui
import click
import logging
import pyqtgraph

@click.command()
@click.option('--config', default='default',
              help="specify the path of configuration file. If use default, the default config file will be used")
def main(config):
    import qdarkstyle
    sys.path.append(str(Path(__file__).parent.parent))
    sys.path.append(str(Path(__file__).parent.parent / 'gui' / 'widgets'))
                           
    app = QApplication(sys.argv)
    
    if config=='default':
        myWin = smartShapeGui()
    else:
        import os
        if os.path.isfile(config):
            print(config)
            myWin = smartShapeGui(config=config)
        else:
            print('The provided config file is not existing! Use default config instead!')
            myWin = smartShapeGui()
    # myWin.init_taurus()
    #TaurusMainWindow.loadSettings(myWin)
    # myWin.loadSettings()
    myWin.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent / 'smart_logo.png')))
    myWin.setWindowTitle("SMART")
    myWin.showMaximized() 
    setattr(myWin, 'app', app)
    #disable warning message
    logging.getLogger('TaurusRootLogger').setLevel(logging.CRITICAL)
    myWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()