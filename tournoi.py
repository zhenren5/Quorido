import numpy as np
import random
import copy, os, time, sys, random, time
G = None          # graph of board (used for connectivity checks)
def initWeights(nb_rows, nb_columns):
    """Fonction destinée à initialiser les poids d'une matrice de genre (nb_rows * nb_columns) pour un réseau de neurones sur base d'une distribution normale de moyenne 0 et d'un écart-type de 0.0001."""
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    """Fonction permettant de créer un réseau de neurones en fonction de la taille de la couche d'entrée (n_input) et de la couche intermédiaire (n_hidden)
    Le réseau de neurones créé est ensuite retourné sous la forme d'un tuple de 2 numpy array contenant respectivement
    les coefficients de poids liant la couche d'entrée à la couche intermédiaire et les coefficients de poids liant la couche intermédiaire à la couche de sortie.
    """
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:,
            0]  # W2 est traité comme vecteur est non comme une matrice Hx1 (simplifie certaines ecritures)
    return (W_int, W_out)


def forwardPass(s, NN, activation):
    """Cette fonction permet d'utiliser un réseau de neurones NN pour estimer la probabilité de victoire finale du joueur blanc pour un état (s) donné du jeu."""
    W_int = NN[0]
    W_out = NN[1]
    P_int = activ(np.dot(W_int, s), None, False, activation)
    p_out = activ(P_int.dot(W_out), None, False, activation)
    return p_out


def backpropagation(s, NN, delta, activation, learning_strategy=None):
    """Fonction destinée à réaliser la mise à jour des poids d'un réseau de neurones (NN). Cette mise à jour se fait conformément à une stratégie d'apprentissage (learning_strategy)
    pour un état donné (s) du jeu.
    Le delta est la différence de probabilité de gain estimée entre deux états successif potentiels du jeu.
    La stratégie d'apprentissage peut soit être None, soit il s'agit d'un tuple de la forme ('Q-learning', alpha) où alpha est le learning_rate (une valeur entre 0 et 1 inclus),
    soit il s'agit d'un tuple de la forme ('TD-lambda', alpha, lamb, Z_int, Z_out) où alpha est le learning_rate, lamb est la valeur de lambda (entre 0 et 1 inclus) et
    Z_int et Z_out contiennent les valeurs de l'éligibility trace associées respectivement aux différents poids du réseau de neurones.
    La fonction de backpropagation ne retourne rien de particulier (None) mais les poids du réseau de neurone NN (W_int, W_out) peuvent être modifiés,
    idem pour l'eligibility trace (Z_int et Z_out) dans le cas où la stratégie TD-lambda est utilisée.
    """
    # remarque : les operateurs +=, -=, et *= changent directement un numpy array sans en faire une copie au prealable, ceci est nécessaire
    # lorsqu'on modifie W_int, W_out, Z_int, Z_out ci-dessous (sinon les changements seraient perdus après l'appel)
    if learning_strategy is None:
        # pas de mise à jour des poids
        return None
    W_int = NN[0]
    W_out = NN[1]
    P_int = activ(np.dot(W_int, s), None, False, activation)
    p_out = activ(P_int.dot(W_out), None, False, activation)
    grad_int = activ(np.dot(W_int, s), P_int, True, activation)
    grad_out = activ(P_int.dot(W_out), p_out, True, activation)
    Delta_int = grad_out * W_out * grad_int
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha * delta * np.outer(Delta_int, s)
        W_out -= alpha * delta * grad_out * P_int
    elif learning_strategy[0] == 'TD-lambda' or learning_strategy[0] == 'Q-lambda':
        alpha = learning_strategy[1]
        lamb = learning_strategy[2]
        Z_int = learning_strategy[3]
        Z_out = learning_strategy[4]
        Z_int *= lamb
        Z_int += np.outer(Delta_int, s)
        # remarque : si au lieu des deux operations ci-dessus nous avions fait
        #    Z_int = lamb*Z_int + np.outer(Delta_int,s)
        # alors cela aura créé un nouveau numpy array, en particulier learning_strategy[3] n'aurait pas été modifié
        Z_out *= lamb
        Z_out += grad_out * P_int
        W_int -= alpha * delta * Z_int
        W_out -= alpha * delta * Z_out


def activ(x, p, derive, activation):
    if activation == "sigmoïde":
        """ calcule la valeur de la fonction sigmoide (de 0 à 1) pour un nombre réel donné.
            Il est à noter que x peut également être un vecteur de type numpy array, dans ce cas, la valeur de la sigmoide correspondante à chaque réel du vecteur est calculée.
            En retour de la fonction, il peut donc y avoir un objet de type numpy.float64 ou un numpy array."""
        if derive:
            res = p * (1 - p)
        else:
            res = 1 / (1 + np.exp(-x))
    elif activation == "ReLU":
        if derive:
            res = 1. * (x > 0)
        else:
            res = np.maximum(0, x)
    elif activation == "tanh":
        if derive:
            res = 1 - p ** 2
        else:
            res = np.tanh(x)
    elif activation=="swish":
        if derive:
            sig=p * (1 - p)
            res=x*sig+sig
        else:
            res = 1 / (1 + np.exp(-x))
    elif activation == "LeakyReLU":
        if derive:
            res = 0.1 * (x > 0)
        else:
            res = np.maximum(0.1 * x, x)
    else:
        res = x
    return res



def makeMove(moves, s, color, NN, eps,activation, learning_strategy=None):
    """Fonction appelée pour que l'IA choisisse une action (le nouvel état new_s est retourné)
     à partir d'une liste d'actions possibles "moves" (c'est-à-dire une liste possible d'états accessibles) à partir d'un état actuel (s)
     Un réseau de neurones (NN) est nécessaire pour faire ce choix quand le mouvement n'est pas du hasard.
     Dans le cas greedy (non aléatoire), la couleur du joueur dont c'est le tour sera utilisée pour déterminer s'il faut retenir le meilleur ou le pire coup du point de vue du joueur blanc.
     Le cas greedy survient avec une probabilité 1-eps.
     La stratégie d'apprentissage fait référence à la stratégie utilisée dans la fonction de backpropagation (cfr la fonction de backpropagation pour une description)
     """
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-lambda')
    # Epsilon greedy
    greedy = random.random() > eps
    best_value = None
    # dans le cas greedy, on recherche le meilleur mouvement (état) possible. Dans le cas du Q-learning (même sans greedy), on a besoin de connaître
    # la probabilité estimée associée au meilleur mouvement (état) possible en vue de réaliser la backpropagation.
    if greedy or Q_learning or Q_lambda:
        best_moves = []
        c = 1
        if color == 1:
            # au cas où c'est noir qui joue, on s'interessera aux pires coups du point de vue de blanc
            c = -1
        for m in moves:
            val = forwardPass(m, NN,activation)
            if best_value == None or c*val > c*best_value: #si noir joue, c'est comme si on regarde alors si val < best_value
                best_moves = [ m ]
                best_value = val
            elif val == best_value:
                best_moves.append(m)
    if greedy:
        # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
        new_s = best_moves[ random.randint(0, len(best_moves)-1) ]
    else:
        # on choisit un mouvement au hasard
        new_s = moves[ random.randint(0, len(moves)-1) ]
    # on met à jour les poids si nécessaire
    if Q_learning or TD_lambda or Q_lambda:
        p_out_s = forwardPass(s, NN,activation)
        if Q_learning:
            delta = p_out_s - best_value
        elif TD_lambda or Q_lambda:
            if greedy:
                p_out_new_s = best_value
            else:
                p_out_new_s = forwardPass(new_s, NN,activation)
            delta = p_out_s - p_out_new_s
        backpropagation(s, NN, delta,activation, learning_strategy)
        if learning_strategy == 'Q-lambda' and p_out_new_s != best_value:
            learning_strategy[3].fill(0)  # remet Z_int à 0
            learning_strategy[4].fill(0)  # remet Z_out à 0
    return new_s

def endGame(s, won, NN, activation, learning_strategy):
    """Cette fonction est appelée en fin de partie, pour une dernière mise à jour des poids lorsqu'une stratégie d'apprentissage est appliquée.
    Les arguments s, NN et learning strategy sont les mêmes que ceux de la fonction makeMove() décrite ci-dessus.
    Ce qui change, c'est le booléen "won" indiquant si le joueur blanc a gagné (True) ou perdu (False). On est ici certain de la probabilité de gain de blanc (1 ou 0).
    Comme précédemment, si une stratégie d'apprentissage est appliquée, un delta sera calculé : ce sera ici sur la base de la probabilité associée au dernier état
    et de la conséquence associée (victoire ou défaite de blanc).
    Il servira à la mise à jour des poids du réseau de neurones.
    Dans le cas du TD-lambda, les éligibility traces sont aussi remises à 0 à la fin.
    Donc, cette fonction ne retourne rien (None) mais elle peut avoir un impact sur les valeurs des poids de NN
    et sur l'eligibility trace associée (quand TD-lambda est utilisée)
    """
    Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
    TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
    Q_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'Q-lambda')
    # on met à jour les poids si nécessaire
    if Q_learning or TD_lambda or Q_lambda:
        p_out_s = forwardPass(s, NN, activation)
        delta = p_out_s - won
        backpropagation(s, NN, delta, activation, learning_strategy)
        if TD_lambda or Q_lambda:
            # on remet les eligibility traces à 0 en prévision de la partie suivante
            learning_strategy[3].fill(0)  # remet Z_int à 0
            learning_strategy[4].fill(0)  # remet Z_out à 0

class Player_AI():
    def __init__(self, NN, eps, learning_strategy, activation,name='IA', N = 5, WALLS = 3):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0
        self.NN = NN
        self.eps = eps
        self.activation = activation
        self.learning_strategy = learning_strategy
        self.N=N
        self.WALLS=WALLS
    def makeMove(self, board):
        return makeMove(self.listMoves(board, self.color), board, self.color, self.NN, self.eps, self.activation, self.learning_strategy)

    def listMoves(self,board, current_player):
        global G
        N=self.N
        WALLS=self.WALLS
        if current_player not in [0, 1]:
            print('error in function listMoves: current_player =', current_player)
        pn = current_player
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(len(steps)):
            steps[i] = np.array(steps[i])
        moves = []
        pos = [None, None]
        coord = [None, None]
        for i in range(2):
            pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
            coord[i] = np.array([pos[i] % N, pos[i] // N])
            pos[i] += pn * N ** 2  # offset for black player
        P = []  # list of new boards (each encoded as list bits to switch)
        # current player moves to another position
        for s in steps:
            if canMove(board, coord[pn], s, N, WALLS):
                new_coord = coord[pn] + s
                new_pos = pos[pn] + s[0] + N * s[1]
                occupied = np.array_equal(new_coord, coord[(pn + 1) % 2])
                if not occupied:
                    P.append([pos[pn], new_pos])  # new board is obtained by switching these two bits
                else:
                    can_jump_straight = canMove(board, new_coord, s, N, WALLS)
                    if can_jump_straight:
                        new_pos = new_pos + s[0] + N * s[1]
                        P.append([pos[pn], new_pos])
                    else:
                        if s[0] == 0:
                            D = [(-1, 0), (1, 0)]
                        else:
                            D = [(0, -1), (0, 1)]
                        for i in range(len(D)):
                            D[i] = np.array(D[i])
                        for d in D:
                            if canMove(board, new_coord, d, N, WALLS):
                                final_pos = new_pos + d[0] + N * d[1]
                                P.append([pos[pn], final_pos])
                                # current player puts down a wall
        # TO DO: Speed up this part: it would perhaps be faster to directly discard intersecting walls based on existing ones
        nb_walls_left = board[
                        2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (pn + 1) * (
                                WALLS + 1)].argmax()
        ind_walls_left = 2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1) + nb_walls_left
        if nb_walls_left > 0:
            for i in range(2 * (N - 1) ** 2):
                pos = 2 * N ** 2 + i
                L = [pos]  # indices of walls that could intersect
                if i < (N - 1) ** 2:
                    # horizontal wall
                    L.append(pos + (N - 1) ** 2)  # vertical wall on the same 4-square
                    if i % (N - 1) > 0:
                        L.append(pos - 1)
                    if i % (N - 1) < N - 2:
                        L.append(pos + 1)
                else:
                    # vertical wall
                    L.append(pos - (N - 1) ** 2)  # horizontal wall on the same 4-square
                    if (i - (N - 1) ** 2) // (N - 1) > 0:
                        L.append(pos - (N - 1))
                    if (i - (N - 1) ** 2) // (N - 1) < N - 2:
                        L.append(pos + (N - 1))
                nb_intersecting_wall = sum([board[j] for j in L])
                if nb_intersecting_wall == 0:
                    board[pos] = 1
                    # we remove the corresponding edges from G
                    if i < (N - 1) ** 2:
                        # horizontal wall
                        a, b = i % (N - 1), i // (N - 1)
                        E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                    else:
                        # vertical wall
                        a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                        E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                    for e in E:
                        G[e[0]][e[1]][e[2]] = 0
                    if self.eachPlayerHasPath(board):
                        P.append(
                            [pos, ind_walls_left - 1, ind_walls_left])  # put down the wall and adapt player's counter
                    board[pos] = 0
                    # we add back the two edges in G
                    for e in E:
                        G[e[0]][e[1]][e[2]] = 1
                        # we create the new boards from P
        for L in P:
            new_board = board.copy()
            for i in L:
                new_board[i] = not new_board[i]
            moves.append(new_board)

        return moves

    def eachPlayerHasPath(self,board):
        global G
        N=self.N
        # heuristic when at most one wall
        nb_walls = board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2].sum()
        if nb_walls <= 1:
            # there is always a path when there is at most one wall
            return True
            # checks whether the two players can each go to the opposite side
        pos = [None, None]
        coord = [[None, None], [None, None]]
        for i in range(2):
            pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
            coord[i][0] = pos[i] % N
            coord[i][1] = pos[i] // N
            coord[i] = np.array(coord[i])
        steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i in range(len(steps)):
            steps[i] = np.array(steps[i])
        for i in range(2):
            A = np.zeros((N, N), dtype='bool')  # TO DO: this could be optimized
            S = [coord[i]]  # set of nodes left to treat
            finished = False
            while len(S) > 0 and not finished:
                c = S.pop()
                # NB: In A we swap rows and columns for simplicity
                A[c[1]][c[0]] = True
                for k in range(4):
                    if G[c[0]][c[1]][k] == 1:
                        s = steps[k]
                        new_c = c + s
                        # test whether we reached the opposite row
                        if i == 0:
                            if new_c[1] == N - 1:
                                finished = True
                                break
                        else:
                            if new_c[1] == 0:
                                return True
                        # otherwise we continue exploring
                        if A[new_c[1]][new_c[0]] == False:
                            # heuristic, we give priority to moves going up (down) in case player is white (black)
                            if i == 0:
                                if k == 1:
                                    S.append(new_c)
                                else:
                                    S.insert(0, new_c)
                            else:
                                if k == 3:
                                    S.append(new_c)
                                else:
                                    S.insert(0, new_c)
            if not finished:
                return False
        return True
    def endGame(self, board, won):
        endGame(board, won, self.NN, self.activation, self.learning_strategy)

def canMove(board, coord, step,N,WALLS):
    # returns True if there is no wall in direction step from pos, and we stay in the board
    # NB: it does not check whether the destination is occupied by a player
    new_coord = coord + step
    in_board = new_coord.min() >= 0 and new_coord.max() <= N - 1
    if not in_board:
        return False
    if WALLS > 0:
        if step[0] == -1:
            L = []
            if new_coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (new_coord[1] - 1) * (N - 1) + new_coord[0])
        elif step[0] == 1:
            L = []
            if coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (coord[1] - 1) * (N - 1) + coord[0])
        elif step[1] == -1:
            L = []
            if new_coord[0] < N - 1:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[0] > 0:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0] - 1)
        elif step[1] == 1:
            L = []
            if coord[0] < N - 1:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[0] > 0:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0] - 1)
        else:
            print('step vector', step, 'is not valid')
            quit(1)
        if sum([board[j] for j in L]) > 0:
            # move blocked by a wall
            return False
    return True



def progressBar(i, n):
    if int(100*i/n) > int(100*(i-1)/n):
        print('  ' + str(int(100*i/n))+'%', end='\r')

class Main():
    """
    Cette classe permet de faire des traitement sur la IA
    """
    def __init__(self,N=5,WALLS=3,EPS=0.2,ALPHA=0.4,LAMB=0.9,LS= 'Q-learning',G_INIT=None,activation="ReLU"):
        """
        initialisation des variables
        window: la classe MyWindow()du fichier partie3.py
        """
        self.NN = None
        self.N = N  # N*N board
        self.WALLS = WALLS   # number of walls each player has
        self.EPS = EPS  # epsilon in epsilon-greedy
        self.ALPHA = ALPHA  # learning_rate
        self.LAMB =LAMB  # lambda for TD(lambda)
        self.LS = LS  # default learning strategy
        self.G_INIT = G_INIT # graph of starting board
        self.activation = activation

    def newIA(self,taille,mur,neurone,eps,alpha,lamb,ls,activation):
        """
        création d'une nouvelle IA avec les paramètres donnés
        """
        self.N = int(taille)
        self.WALLS = int(mur)
        self.NN = createNN(2 * self.N ** 2 + 2 * (self.N - 1) ** 2 + 2 * (self.WALLS + 1), int(neurone))
        self.EPS = float(eps)
        self.ALPHA = float(alpha)
        self.LAMB = float(lamb)
        self.activation=activation
        self.LS = ls
        self.G_INIT = self.computeGraph()

    def save(self, filename):
        """
        sauvegarde IA dans le fichier filename, et retourne le text à afficher sur la fenêtre
        """
        np.savez(filename, N=self.N, WALLS=self.WALLS, W1=self.NN[0], W2=self.NN[1])
    def train(self,n_train):
        """
        créer des joueurs IA en fonction de stratégie choisie par l'utilisateur
        retourne un tuple (agent1,agent2) qui permet ensuite à entraîner
        """
        eps_min=0.1
        decay=0.99999#527811
        if self.LS == 'Q-learning':
            learning_strategy1 = (self.LS, self.ALPHA)
            learning_strategy2 = (self.LS, self.ALPHA)
        else:# stratégie 'TD-lambda' ou 'Q_lambda'
            learning_strategy1 = (self.LS, self.ALPHA, self.LAMB, np.zeros(self.NN[0].shape), np.zeros(self.NN[1].shape))
            learning_strategy2 = (self.LS, self.ALPHA, self.LAMB, np.zeros(self.NN[0].shape), np.zeros(self.NN[1].shape))
        agent1 = Player_AI(self.NN, self.EPS, learning_strategy1,self.activation,'agent 1', N=self.N, WALLS=self.WALLS)
        agent2 = Player_AI(self.NN, self.EPS, learning_strategy2,self.activation,'agent 2', N=self.N, WALLS=self.WALLS)
        print('\nEntraînement (' + str(n_train) + ' parties)')
        for j in range(n_train):
            progressBar(j, n_train)
            self.playGame(agent1, agent2)
            eps = agent1.eps
            if eps>eps_min:
                agent1.eps=max(eps_min,agent1.eps*decay)
                agent2.eps=max(eps_min,agent2.eps*decay)
        print('\nEntraînement terminé')

    def computeGraph(self, board=None):
        # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
        pos_steps = [(1, 0), (0, 1)]
        for i in range(len(pos_steps)):
            pos_steps[i] = np.array(pos_steps[i])
        g = np.zeros((self.N, self.N, 4))
        for i in range(self.N):
            for j in range(self.N):
                c = np.array([i, j])
                for k in range(2):
                    s = pos_steps[k]
                    if board is None:
                        # initial setup
                        new_c = c + s
                        if new_c.min() >= 0 and new_c.max() <= self.N - 1:
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
                    else:
                        if canMove(board, c, s, self.N, self.WALLS):
                            new_c = c + s
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
        return g

    def endOfGame(self,board):
        N=self.N
        return board[(N - 1) * N:N ** 2].max() == 1 or board[N ** 2:N ** 2 + N].max() == 1
    def playGame(self,player1, player2):
        global G
        # initialization
        players = [player1, player2]
        board = self.startingBoard()
        G = self.G_INIT.copy()
        N=self.N
        for i in range(2):
            players[i].color = i
        # main loop
        finished = False
        current_player = 0
        count = 0
        quit = False
        while not finished:
            new_board = players[current_player].makeMove(board)
            # we compute changes of G (if any) to avoid recomputing G at beginning of listMoves
            # we remove the corresponding edges from G
            if not new_board is None:
                v = new_board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2] - board[
                                                                          2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2]
                i = v.argmax()
                if v[i] == 1:
                    # a wall has been added, we remove the two corresponding edges of G
                    if i < (N - 1) ** 2:
                        # horizontal wall
                        a, b = i % (N - 1), i // (N - 1)
                        E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                    else:
                        # vertical wall
                        a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                        E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                    for e in E:
                        G[e[0]][e[1]][e[2]] = 0
            board = new_board
            if board is None:
                # human player quit
                quit = True
                finished = True
            elif self.endOfGame(board):
                players[current_player].score += 1
                white_won = current_player == 0
                players[current_player].endGame(board, white_won)
                finished = True
            else:
                current_player = (current_player + 1) % 2
        return quit

    def startingBoard(self):
        N=self.N
        WALLS=self.WALLS
        board = np.array([0] * (2 * N ** 2 + 2 * (N - 1) ** 2 + 2 * (WALLS + 1)))
        # player positions
        board[(N - 1) // 2] = True
        board[N ** 2 + N * (N - 1) + (N - 1) // 2] = True
        # wall counts
        for i in range(2):
            board[2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1) + WALLS] = 1
        return board

n_train,filename=sys.argv[1:]
m=Main()
m.newIA(5,3,40,0.7,0.4,0.9,'Q-learning',"LeakyReLU")
print("LRELU")
m.train(int(n_train))
m.save(filename)