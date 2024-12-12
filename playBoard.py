"""
nom:Ren  prénom:Zheng 
une sous classe de widget qui permet de dessiner le plateau du jeu
"""
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QColor,QPen
from PyQt5.QtCore import Qt
from main_partie4 import *

class Board(QtWidgets.QWidget):
    def __init__(self, parent,size,walls):
        QtWidgets.QWidget.__init__(self, parent)
        self.grid = QtWidgets.QGridLayout()
        self.boardsize=size # taille du plateau
        self.walls_p1=walls # nombre de mur pour le premier joueur
        self.walls_p2=walls # nombre de mur pour le deuxieme joueur
        self.player2_pos = [(self.boardsize // 2), self.boardsize - 1] # position du deuxieme joueur [colonne,ligne]
        self.player1_pos = [(self.boardsize // 2), 0] # position du premier joueur [colonne,ligne]
        self.w_hor=[] #une liste de position des murs horizontals sous forme(colonne,ligne)
        self.w_ver=[] #une liste de position des murs vertials sous forme(colonne,ligne)

    def set_size(self,size):
        """
        changement de taille et position des pions
        :param size: la nouvelle taille du plateau
        """
        self.boardsize=size
        self.player2_pos = [(self.boardsize // 2),self.boardsize - 1]
        self.player1_pos = [(self.boardsize // 2), 0]
    def set_walls(self,walls):
        self.walls_p1 = walls
        self.walls_p2 = walls
    def paintEvent(self, e):
        """dessine le plateau du jeu, les pions et les murs"""
        qp = QPainter()
        qp.begin(self)
        self.board(qp)
        qp.end()
    def board(self,qp):
        """
        dessine le plateau du jeu
        """
        width = self.width()
        height = self.height()
        length = min(width, height) // (self.boardsize + 2)
        dimension=length * self.boardsize
        x = (width - dimension) // 2
        y = height-(height - dimension) // 2
        p1=self.player1_pos
        p2=self.player2_pos
        for i in range(self.boardsize):
            for j in range(self.boardsize,0,-1):
                self.square(qp, x + i * length, y - j * length, length)
        self.piece(qp, Qt.white, x + length *p1[0], y -(p1[1]+1)*length, length)#on compte de bas en haut
        self.piece(qp, Qt.black, x + length *p2[0],y -(p2[1]+1)*length, length)
        for i in range(self.walls_p2):
            self.wall_left(qp, x + i * 10, 0, length)
        for i in range(self.walls_p1):
            self.wall_left(qp, x + i * 10, height-length, length)
        self.wall(qp,x,y,length)

    def square(self, qp,x,y,length):
        """
        dessine un carré avec (x,y) le point supérieur à gauche et de longueur length
        """
        qp.setPen(Qt.black)
        qp.setBrush(QColor(123,240,153))
        qp.drawRect(x, y, length, length)

    def piece(self,qp,color,x,y,d):
        """
        dessine le pion
        :param color: ls couleur du pion
        (x,y) le point supérieur à gauche du pion
        :param d: diamètre du pion
        """
        qp.setPen(Qt.black)
        qp.setBrush(color)
        qp.drawEllipse(x,y,d,d)
    def wall_left(self,qp,x,y,length):
        """
        dessine les murs restants avec (x,y)le point de départ et length longueur à dessiner
        """
        qp.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        qp.drawLine(x, y, x, y + length)
    def wall(self,qp,x,y,length):
        """
        dessine les murs horizontals et verticals dans le plateau
        (x,y)le coin supérieur à gauche du plateau
        length: la logueur de chaque case
        """
        qp.setPen(QPen(Qt.red,  4, Qt.SolidLine))
        for i in range(len(self.w_hor)):
            k,l=self.w_hor[i]
            x1=x+k*length
            qp.drawLine(x1, y-(l+1)*length, x1+2*length, y-(l+1)*length)
        for i in range(len(self.w_ver)):
            k, l = self.w_ver[i]
            y1 = y - l * length
            qp.drawLine(x + (k + 1) * length, y1, x + (k + 1) * length, y1-2*length)


    def display(self,board,window, msg=None):
        """
        mis à jour des nouvelles donées grâce aux positions données par la fonction listEncoding
        et redessine le plateau
        """
        lboard =listEncoding(board,window)
        self.player1_pos=lboard[0]
        self.player2_pos=lboard[1]
        self.w_hor=lboard[2]
        self.w_ver=lboard[3]
        self.walls_p1=lboard[-2]
        self.walls_p2=lboard[-1]
        self.repaint()
        if not msg is None:
            window.ui.textEdit.setText(msg)
