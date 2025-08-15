# -*- coding: utf-8 -*-


# // module to manage the field view
# from ui.workspace_widget import Ui_workspace_widget
import sys, socket
import yaml
from pathlib import Path
import numpy as np
import qdarkstyle
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QMainWindow
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import pyqtSlot as Slot, QTimer
from PyQt5.QtWidgets import QAction, QToolBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from .widgets.shape_viewer_widget_control import viewerControl
from smart_shape import icon_path

setting_file = str(
    Path(__file__).parent.parent / "config" / "app_config.yml"
)
ui_file_folder = Path(__file__).parent / "ui"

class smartShapeGui(QMainWindow, viewerControl):
    """
    Main class of the workspace
    """

    statusMessage_sig = Signal(str)

    def __init__(self, parent=None, config="default"):
        """
        Initialize the class
        :param parent: parent widget
        :param settings_object: settings object
        """
        super().__init__(parent=parent)
        self.__init_gui(config=config)
        self.user_right = "normal"
        self.add_smart_toolbar()
        self.widget_pars.init_pars(self.settings_object)

    def __init_gui(self, config):
        uic.loadUi(str(ui_file_folder / "smart_shape_main_window.ui"), self)
        if config == "default":
            # self.settings_object = QtCore.QSettings(setting_file, QtCore.QSettings.IniFormat)
            self.setting_file_yaml = str(setting_file)
            with open(str(setting_file), "r", encoding="utf8") as f:
                self.settings_object = yaml.safe_load(f.read())
        else:
            self.setting_file_yaml = str(config)
            with open(str(config), "r", encoding="utf8") as f:
                self.settings_object = yaml.safe_load(f.read())
            # self.settings_object = QtCore.QSettings(config, QtCore.QSettings.IniFormat)

        self._init_viewer()
        self.widget_terminal.update_name_space("gui", self)
        self._parent = self

        self.connect_slots()
        self.init_attribute_values()

    def change_stylesheet(self, on_or_off, action_icons):
        if hasattr(self, 'app'):
            app = self.app
            if on_or_off:
                app.setStyleSheet('')
            else:
                app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        str_mode = ['k','w'][int(on_or_off)]
        if on_or_off:
            action_icons[0].setVisible(False)
            action_icons[1].setVisible(True)
        else:
            action_icons[1].setVisible(False)
            action_icons[0].setVisible(True)

    def add_smart_toolbar(self):
        tb = QToolBar("SMART Toolbar")
        tb.setObjectName("SMART Toolbar")
        #        tb.addAction(self.changeTangoHostAction)
        #        tb.addWidget(self.taurusLogo)
        lighton = QAction(QIcon(str(icon_path / 'smart' / 'lighton.png')),'turn on light',self)
        lighton.setStatusTip('Change the GUI stylesheet to light mode.')
        lighton.setVisible(False)
        lightoff = QAction(QIcon(str(icon_path / 'smart' / 'lightoff.png')),'turn off light',self)
        lightoff.setStatusTip('Change the GUI stylesheet to dark mode.')
        lighton.triggered.connect(lambda: self.change_stylesheet(True,[lighton,lightoff]))        
        lightoff.triggered.connect(lambda: self.change_stylesheet(False,[lighton,lightoff]))  
        self.change_stylesheet(True,[lighton,lightoff])
        lightoff.setVisible(True)
        tb.addAction(lighton)
        tb.addAction(lightoff)
        self.smart_toolbar = tb
        self.addToolBar(self.smart_toolbar)

    def init_attribute_values(self):
        #TODO: this func should be get rid of, since these attributes are synoptic specific for psd pump, should not be set as global ones
        self.lines_draw_before = []
        self.lines_draw_after = []
        self.pen_lines_draw_before = []
        self.pen_lines_draw_after = []

    def connect_slots(self):
        """
        :return:
        """
        # synoptic viewer slots
        self.widget_synoptic.connect_slots_synoptic_viewer()
        self.connect_slots_synoptic_viewer_control()

    def statusUpdate(self, m):
        # slot for showing a message in the statusbar.
        self.statusbar.showMessage(m)

    def closeEvent(self, event):
        quit_msg = "About to Exit the program, are you sure? "
        reply = QMessageBox.question(
            self, "Message", quit_msg, QMessageBox.Yes, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        elif reply == QMessageBox.No:
            event.ignore()
