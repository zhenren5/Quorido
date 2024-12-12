"""
nom:Ren  prénom:Zheng 
Partie 4 du projet d’année: amélioration de IA
#NOUVELLE METHODE: La stratégie d’apprentissage Q(λ) de Watkins et les fonctions d’activation
"""
from PyQt5.QtWidgets import QMessageBox,QApplication
import partie4_qt
from main_partie4 import *
from playBoard import *
import sys

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = partie4_qt.Ui_MainWindow()
        self.ui.setupUi(self)
        self.p3 = Main(self)
        self.ui.saveButton.clicked.connect(self.saveFile)
        self.ui.createButton.clicked.connect(self.create)
        self.ui.loadButton.clicked.connect(self.loadFile)
        self.ui.comparepushButton.clicked.connect(self.compare_IA)
        self.ui.trainButton.clicked.connect(self.trainIA)
        self.ui.playButton.clicked.connect(self.playBtn)
        self.boardW=Board(self.ui.centralwidget, self.ui.sizespinBox.value(),self.ui.nbrWallpinBox.value())
        self.ui.boardWidget=self.boardW # un widget qui dessine le plateau du jeu
        self.ui.boardWidget.setObjectName("boardWidget")
        self.ui.gridLayout_2.addWidget(self.ui.boardWidget, 0, 2, 1, 1)
        self.isplay=False # un booléan qui permet de indiquer si le joueur humain est en train de jouer
        self.key=None # donne la touche de clavier que le joueur a appuyé
        self.setFocus(Qt.TabFocusReason)
        self.setFocusPolicy(Qt.StrongFocus)
        self.boardW.hide()
        self.finish=False
    def create(self):
        """
        fonction liée au bouton createButton
        permet la création d'un nouveau IA en fonctin des paramètres indiqués par l'utilisateur
        """
        self.boardW.hide()
        self.ui.progressBar.setValue(0)
        taille=self.ui.sizespinBox.value()
        mur=self.ui.nbrWallpinBox.value()
        neurone=self.ui.neuronespinBox.value()
        eps=self.ui.epsilonSpinBox.value()
        alpha=self.ui.learningRateSpinBox.value()
        lamb=self.ui.lambdaSpinBox.value()
        ls=self.ui.strategycomboBox.currentText()
        self.p3.newIA(taille,mur,neurone,eps,alpha,lamb,ls)
        self.ui.textEdit.setText(self.p3.info(True))
        self.boardW.set_size(taille)
        self.boardW.set_walls(mur)
    def saveFile(self):
        """
        cette fonction permet de sauvegarder IA dans le fichier que le nom est donné par l'utilisateur,
        liée au bouton saveButton
        """
        self.boardW.hide()
        self.ui.progressBar.setValue(0)
        self.ui.textEdit.setText(self.p3.save(self.ui.savelineEdit.text()))
    def loadFile(self):
        """
        cette fonction permet de charger une IA du fichier que le nom est donné par l'utilisateur,
        liée au bouton loadButton
        """
        self.boardW.hide()
        self.ui.progressBar.setValue(0)
        txt=self.p3.load(self.ui.loadlineEdit.text())
        if(txt==""):
            txt=self.p3.info(False)
        self.ui.textEdit.setText(txt)
    def trainIA(self):
        """
        entraînement de IA,liée au bouton trainButton
        """
        self.boardW.hide()
        self.ui.progressBar.setValue(0)
        n_train=self.ui.trainspinBox.value()
        activation = self.ui.activ1comboBox.currentText()
        if self.p3.NN is None:
            self.ui.textEdit.setText("Il faut d'abord créer ou charger une IA")
        else:
            self.ui.textEdit.setText('Entraînement (' + str(n_train) + ' parties)')
            agent1,agent2=self.p3.traine(activation)
            for j in range(n_train):
                self.progressBar(j, n_train)
                playGame(self,agent1, agent2,activation,activation)
                QtWidgets.QApplication.processEvents()
            self.ui.progressBar.setValue(100)
            self.ui.textEdit.append("\n                        Terminé")
    def compare_IA(self):
        """
        comparaison de IA avec IA du fichier filename , liée au bouton comparepushButton
        """
        self.boardW.hide()
        self.ui.progressBar.setValue(0)
        filename=self.ui.comparelineEdit.text()
        n_compare=self.ui.comparespinBox.value()
        eps=self.ui.epsilonSpinBox.value()
        activation1 = self.ui.activ1comboBox.currentText()
        activation2 = self.ui.activ2comboBox.currentText()
        if self.p3.NN is None:
            self.ui.textEdit.setText("Il faut d'abord créer ou charger une IA")
        elif filename == "":
            self.ui.textEdit.setText("veillez donner le nom du fichier à comparer")
        else:
            try:
                self.ui.textEdit.setText('Tournoi IA vs ' + filename + ' (' + str(n_compare) + ' parties, eps=' + str(eps) + ')')
                players = self.p3.compare(filename,eps,activation1,activation2)
                i = 0
                for j in range(n_compare):
                    self.progressBar(j, n_compare)
                    playGame(self,players[i], players[(i + 1) % 2],activation1,activation2)
                    i = (i + 1) % 2
                    QtWidgets.QApplication.processEvents()
                perf = players[0].score / n_compare
                self.ui.textEdit.append("\nParties gagnées par l'IA :" + "{0:.2f}".format(perf * 100) + '%')
                self.ui.progressBar.setValue(100)
                self.ui.textEdit.append("\n                        Terminé")
            except FileNotFoundError:
                self.ui.textEdit.setText("\nERROR!! Le fichier n'existe pas")
            except TypeError:
                self.ui.textEdit.setText("\nERROR!! Mauvais fichier")
            except:
                self.ui.textEdit.setText("\nERROR!! Les deux IA doivent avoir le même nombre de mur et taille!!")
    def playBtn(self):
        """
        cette fonction est liée au boutton playButton, pour jouer le jeu Quoridor
        """
        self.ui.progressBar.setValue(0)
        player1=self.ui.player1comboBox.currentText()
        player2=self.ui.player2comboBox.currentText()
        taille=self.ui.sizespinBox.value()
        mur=self.ui.nbrWallpinBox.value()
        QtWidgets.QApplication.processEvents()
        activation1 = self.ui.activ1comboBox.currentText()
        activation2=self.ui.activ2comboBox.currentText()
        is_IA=self.p3.playQuoridor(player1, player2,taille,mur,activation1,activation2)
        if is_IA:
            self.isplay=False
            human = Player_Human('Humain')
            agent = Player_AI(self.p3.NN, 0.0, None, 'IA')  # IA joue le mieux possible
            if (player1 == "Human" and player2 == "IA"):
                self.disable(False)
                self.boardW.show()
                play(self,human, agent,activation1,activation2)
            elif (player1 == "IA" and player2 == "Human"):
                self.disable(False)
                self.boardW.show()
                play(self,agent, human,activation1,activation2)
            else:
                self.trainIA()
        self.disable(True)
    def progressBar(self,i, n):
        """
        affchage de valeur dans progressbar
        :param i: le nombre de tour a effectué
        :param n: nombre total d'entraînement ou comparaison
        """
        if int(100 * i / n) > int(100 * (i - 1) / n):
            self.ui.progressBar.setValue(100 * i / n)

    def keyPressEvent(self, event):
        """
        évenement déclenché par la touche de clavier, modifie la variable key si le joueur est en train de jouer
        """
        if self.isplay:
            if event.key() == Qt.Key_A:
                self.key = 'a'
            if event.key() == Qt.Key_D:
                self.key = 'd'
            if event.key() == Qt.Key_F:
                self.key = 'f'
            if event.key() == Qt.Key_C:
                self.key = 'c'
            if event.key() == Qt.Key_V:
                self.key = 'v'
            if event.key() == Qt.Key_W:
                self.key = 'w'
            if (event.key() ==Qt.Key_Q):
                self.key='q'
            if (event.key() ==Qt.Key_Right):
                self.key='RIGHT'
            if (event.key() == Qt.Key_Up):
                self.key = "UP"
            if (event.key() == Qt.Key_Down):
                self.key = "DOWN"
            if (event.key() == Qt.Key_Left):
                self.key = "LEFT"
            if event.key() == 16777220 or event.key() == Qt.Key_Enter:
                self.key = '\r'
    def closeEvent(self, event):
        """
        l'évènement décleché lors de la fermeture de la fenêtre
        """
        reply = QMessageBox.question(self, 'Message', 'Vouley-vous vraiment quitter le programme?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    def waitForKey(self):
        """
        cette fonction est appelée lorsque le joueur doit choisir le mouvement du pion
        """
        self.isplay = True
        key = self.key
        while key is None: # attendre jusqu'à le joueur a appuyé une touche acceptable
            QtWidgets.QApplication.processEvents()
            key = self.key
        self.isplay = False
    def disable(self,bol):
        """
        active ou désactive les boutons selon la paramètre bol
        """
        self.ui.trainButton.setEnabled(bol)
        self.ui.saveButton.setEnabled(bol)
        self.ui.comparepushButton.setEnabled(bol)
        self.ui.createButton.setEnabled(bol)
        self.ui.comparepushButton.setEnabled(bol)
        self.ui.loadButton.setEnabled(bol)
        self.ui.playButton.setEnabled(bol)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())