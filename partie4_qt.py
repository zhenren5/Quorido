# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'partie4_qt.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(859, 606)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(204, 240, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(204, 240, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(204, 240, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(204, 240, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setMouseTracking(False)
        self.progressBar.setAutoFillBackground(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_2.addWidget(self.progressBar, 2, 2, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setMaximumSize(QtCore.QSize(6044, 185))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.textEdit.setFont(font)
        self.textEdit.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_2.addWidget(self.textEdit, 1, 2, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.savelineEdit = QtWidgets.QLineEdit(self.widget)
        self.savelineEdit.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.savelineEdit.setObjectName("savelineEdit")
        self.gridLayout.addWidget(self.savelineEdit, 0, 1, 1, 1)
        self.saveButton = QtWidgets.QPushButton(self.widget)
        self.saveButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.saveButton.setObjectName("saveButton")
        self.gridLayout.addWidget(self.saveButton, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.loadlineEdit = QtWidgets.QLineEdit(self.widget)
        self.loadlineEdit.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.loadlineEdit.setObjectName("loadlineEdit")
        self.gridLayout.addWidget(self.loadlineEdit, 1, 1, 1, 1)
        self.loadButton = QtWidgets.QPushButton(self.widget)
        self.loadButton.setMinimumSize(QtCore.QSize(0, 0))
        self.loadButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.loadButton.setObjectName("loadButton")
        self.gridLayout.addWidget(self.loadButton, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.sizespinBox = QtWidgets.QSpinBox(self.widget)
        self.sizespinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.sizespinBox.setMinimum(3)
        self.sizespinBox.setMaximum(9)
        self.sizespinBox.setProperty("value", 5)
        self.sizespinBox.setObjectName("sizespinBox")
        self.gridLayout.addWidget(self.sizespinBox, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.nbrWallpinBox = QtWidgets.QSpinBox(self.widget)
        self.nbrWallpinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.nbrWallpinBox.setMaximum(10)
        self.nbrWallpinBox.setProperty("value", 1)
        self.nbrWallpinBox.setObjectName("nbrWallpinBox")
        self.gridLayout.addWidget(self.nbrWallpinBox, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.epsilonSpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.epsilonSpinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.epsilonSpinBox.setMaximum(1.0)
        self.epsilonSpinBox.setSingleStep(0.01)
        self.epsilonSpinBox.setProperty("value", 0.3)
        self.epsilonSpinBox.setObjectName("epsilonSpinBox")
        self.gridLayout.addWidget(self.epsilonSpinBox, 4, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.learningRateSpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.learningRateSpinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.learningRateSpinBox.setMaximum(1.0)
        self.learningRateSpinBox.setSingleStep(0.01)
        self.learningRateSpinBox.setProperty("value", 0.4)
        self.learningRateSpinBox.setObjectName("learningRateSpinBox")
        self.gridLayout.addWidget(self.learningRateSpinBox, 5, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 6, 0, 1, 1)
        self.lambdaSpinBox = QtWidgets.QDoubleSpinBox(self.widget)
        self.lambdaSpinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.lambdaSpinBox.setMaximum(1.0)
        self.lambdaSpinBox.setSingleStep(0.01)
        self.lambdaSpinBox.setProperty("value", 0.9)
        self.lambdaSpinBox.setObjectName("lambdaSpinBox")
        self.gridLayout.addWidget(self.lambdaSpinBox, 6, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 7, 0, 1, 1)
        self.neuronespinBox = QtWidgets.QSpinBox(self.widget)
        self.neuronespinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.neuronespinBox.setMaximum(500000000)
        self.neuronespinBox.setSingleStep(10)
        self.neuronespinBox.setProperty("value", 40)
        self.neuronespinBox.setObjectName("neuronespinBox")
        self.gridLayout.addWidget(self.neuronespinBox, 7, 1, 1, 1)
        self.nbrentrainementlabel = QtWidgets.QLabel(self.widget)
        self.nbrentrainementlabel.setObjectName("nbrentrainementlabel")
        self.gridLayout.addWidget(self.nbrentrainementlabel, 8, 0, 1, 1)
        self.trainspinBox = QtWidgets.QSpinBox(self.widget)
        self.trainspinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.trainspinBox.setMaximum(999999999)
        self.trainspinBox.setSingleStep(1)
        self.trainspinBox.setProperty("value", 10000)
        self.trainspinBox.setObjectName("trainspinBox")
        self.gridLayout.addWidget(self.trainspinBox, 8, 1, 1, 1)
        self.nbrentrainementlabel_2 = QtWidgets.QLabel(self.widget)
        self.nbrentrainementlabel_2.setObjectName("nbrentrainementlabel_2")
        self.gridLayout.addWidget(self.nbrentrainementlabel_2, 9, 0, 1, 1)
        self.comparespinBox = QtWidgets.QSpinBox(self.widget)
        self.comparespinBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.comparespinBox.setMaximum(999999999)
        self.comparespinBox.setProperty("value", 1000)
        self.comparespinBox.setObjectName("comparespinBox")
        self.gridLayout.addWidget(self.comparespinBox, 9, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 10, 0, 1, 1)
        self.comparelineEdit = QtWidgets.QLineEdit(self.widget)
        self.comparelineEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.comparelineEdit.setObjectName("comparelineEdit")
        self.gridLayout.addWidget(self.comparelineEdit, 10, 1, 1, 1)
        self.comparepushButton = QtWidgets.QPushButton(self.widget)
        self.comparepushButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.comparepushButton.setObjectName("comparepushButton")
        self.gridLayout.addWidget(self.comparepushButton, 10, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 11, 0, 1, 1)
        self.strategycomboBox = QtWidgets.QComboBox(self.widget)
        self.strategycomboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.strategycomboBox.setEditable(False)
        self.strategycomboBox.setObjectName("strategycomboBox")
        self.strategycomboBox.addItem("")
        self.strategycomboBox.addItem("")
        self.strategycomboBox.addItem("")
        self.gridLayout.addWidget(self.strategycomboBox, 11, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.widget)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 12, 0, 1, 1)
        self.player1comboBox = QtWidgets.QComboBox(self.widget)
        self.player1comboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.player1comboBox.setEditable(False)
        self.player1comboBox.setObjectName("player1comboBox")
        self.player1comboBox.addItem("")
        self.player1comboBox.addItem("")
        self.gridLayout.addWidget(self.player1comboBox, 12, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 13, 0, 1, 1)
        self.player2comboBox = QtWidgets.QComboBox(self.widget)
        self.player2comboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.player2comboBox.setEditable(False)
        self.player2comboBox.setObjectName("player2comboBox")
        self.player2comboBox.addItem("")
        self.player2comboBox.addItem("")
        self.gridLayout.addWidget(self.player2comboBox, 13, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 14, 0, 1, 1)
        self.activ1comboBox = QtWidgets.QComboBox(self.widget)
        self.activ1comboBox.setObjectName("activ1comboBox")
        self.activ1comboBox.addItem("")
        self.activ1comboBox.addItem("")
        self.activ1comboBox.addItem("")
        self.activ1comboBox.addItem("")
        self.activ1comboBox.addItem("")
        self.gridLayout.addWidget(self.activ1comboBox, 14, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.widget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 15, 0, 1, 1)
        self.activ2comboBox = QtWidgets.QComboBox(self.widget)
        self.activ2comboBox.setObjectName("activ2comboBox")
        self.activ2comboBox.addItem("")
        self.activ2comboBox.addItem("")
        self.activ2comboBox.addItem("")
        self.activ2comboBox.addItem("")
        self.activ2comboBox.addItem("")
        self.gridLayout.addWidget(self.activ2comboBox, 15, 1, 1, 1)
        self.createButton = QtWidgets.QPushButton(self.widget)
        self.createButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.createButton.setObjectName("createButton")
        self.gridLayout.addWidget(self.createButton, 16, 0, 1, 1)
        self.trainButton = QtWidgets.QPushButton(self.widget)
        self.trainButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.trainButton.setObjectName("trainButton")
        self.gridLayout.addWidget(self.trainButton, 16, 1, 1, 1)
        self.playButton = QtWidgets.QPushButton(self.widget)
        self.playButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.playButton.setObjectName("playButton")
        self.gridLayout.addWidget(self.playButton, 16, 2, 1, 1)
        self.gridLayout_2.addWidget(self.widget, 0, 3, 3, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 859, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", " Quoridor"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Segoe Print\'; font-size:9pt; font-weight:600; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Bienvenue au Quoridor!</p></body></html>"))
        self.label.setText(_translate("MainWindow", "Nom de fichier à sauvegarder"))
        self.saveButton.setText(_translate("MainWindow", "Save file"))
        self.label_2.setText(_translate("MainWindow", "Nom de fichier à charger"))
        self.loadButton.setText(_translate("MainWindow", "Load file"))
        self.label_3.setText(_translate("MainWindow", "Taille du plateau"))
        self.label_4.setText(_translate("MainWindow", "Nombre du murs"))
        self.label_5.setText(_translate("MainWindow", "Epsilon"))
        self.label_6.setText(_translate("MainWindow", "Learning rate"))
        self.label_7.setText(_translate("MainWindow", "Lambda for TD"))
        self.label_8.setText(_translate("MainWindow", "Nombre de neurone sur la couche  intermédiaire "))
        self.nbrentrainementlabel.setText(_translate("MainWindow", "Nombre de tours pour entraîner  l\'IA"))
        self.nbrentrainementlabel_2.setText(_translate("MainWindow", "Nombre de tours pour comparer l\'IA"))
        self.label_9.setText(_translate("MainWindow", "Nom de fichier à comparer"))
        self.comparepushButton.setText(_translate("MainWindow", "Compare"))
        self.label_10.setText(_translate("MainWindow", "IA strategy"))
        self.strategycomboBox.setItemText(0, _translate("MainWindow", "Q-learning"))
        self.strategycomboBox.setItemText(1, _translate("MainWindow", "TD-lambda"))
        self.strategycomboBox.setItemText(2, _translate("MainWindow", "Q-lambda"))
        self.label_13.setText(_translate("MainWindow", "Player 1"))
        self.player1comboBox.setItemText(0, _translate("MainWindow", "IA"))
        self.player1comboBox.setItemText(1, _translate("MainWindow", "Human"))
        self.label_14.setText(_translate("MainWindow", "Player 2"))
        self.player2comboBox.setItemText(0, _translate("MainWindow", "Human"))
        self.player2comboBox.setItemText(1, _translate("MainWindow", "IA"))
        self.label_11.setText(_translate("MainWindow", "Activation pour entraînement/ Activation IA1"))
        self.activ1comboBox.setItemText(0, _translate("MainWindow", "sigmoïde "))
        self.activ1comboBox.setItemText(1, _translate("MainWindow", "tanh"))
        self.activ1comboBox.setItemText(2, _translate("MainWindow", "ReLU"))
        self.activ1comboBox.setItemText(3, _translate("MainWindow", "LeakyReLU"))
        self.activ1comboBox.setItemText(4, _translate("MainWindow", "swish"))
        self.label_12.setText(_translate("MainWindow", "Activation IA2"))
        self.activ2comboBox.setItemText(0, _translate("MainWindow", "sigmoïde "))
        self.activ2comboBox.setItemText(1, _translate("MainWindow", "tanh"))
        self.activ2comboBox.setItemText(2, _translate("MainWindow", "ReLU"))
        self.activ2comboBox.setItemText(3, _translate("MainWindow", "LeakyReLU"))
        self.activ2comboBox.setItemText(4, _translate("MainWindow", "swish"))
        self.createButton.setText(_translate("MainWindow", "Create New IA"))
        self.trainButton.setText(_translate("MainWindow", "Train the IA"))
        self.playButton.setText(_translate("MainWindow", "Play with current parameters"))
