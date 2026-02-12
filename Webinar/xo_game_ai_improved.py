"""
🧠 X și O cu Machine Learning - VERSIUNE INTEGRATĂ
==================================================
Combină UI-ul original cu un sistem de antrenament îmbunătățit.

Caracteristici:
- Interfață grafică prietenoasă construită cu PyQt5.
- Agent de Machine Learning (Q-Learning) care învață să joace.
- Sistem de antrenament avansat cu moduri multiple, inclusiv:
  - Curriculum Learning: Progresie de la adversari ușori la experți.
  - Experience Replay: Îmbunătățește stabilitatea învățării.
  - Epsilon Decay: Echilibrează explorarea cu exploatarea.
- Viteză de antrenament optimizată prin rularea pe un thread separat.
- Posibilitatea de a salva și încărca progresul modelului AI.
- Vizualizarea în timp real a statisticilor și a procesului de învățare.
- Compatibilitate cu modelele vechi și noi.

Autori: Lao & Claude - Webinar "Arta Programării cu AI"
"""



# ═══════════════════════════════════════════════════════════════════════
# 🧠 SECȚIUNEA DE ÎNVĂȚARE - CUM FUNCȚIONEAZĂ AI-UL - Reinforcement Learning
# ═══════════════════════════════════════════════════════════════════════
#
# Q-table este efectiv modelul matematic prin care algoritmul învață.
# Q-table = o memorie matematică de forma (state, action) → valoare
# „Dacă sunt în această stare a tablei și fac această mutare, cât de bună este mutarea?”
#
# Procesul de învățare în 5 pași:
# 1️⃣ AI-ul joacă un joc complet (9 mutări maxim)
# 2️⃣ La final, primește scor: +1 (câștig), -1 (pierdere), +0.1 (egal)
# 3️⃣ Fiecare mutare din joc primește o parte din acest scor
# 4️⃣ AI-ul își actualizează "memoria" (Q-table) cu ce a învățat
# 5️⃣ La jocul următor, alege mutările cu scorurile cele mai mari
#
# Exemplu concret:
# - Mutarea "pune O în colț când tabla e goală" → după 100 jocuri → scor 0.85
# - Mutarea "pune O pe margine când tabla e goală" → după 100 jocuri → scor 0.23
# - AI-ul alege 0.85 > 0.23 → preferă colțul!
#
# După 1000+ jocuri, AI-ul "știe" ce mutări duc la victorie! 🎯
# ═══════════════════════════════════════════════════════════════════════



import sys
import random
import pickle
import numpy as np
from collections import defaultdict
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QDialog, QTextEdit,
                             QGridLayout, QSplitter, QProgressBar,
                             QComboBox, QSpinBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


# -----------------------------------------------------------------------------
# CLASELE DE BAZĂ PENTRU LOGICA AI
# -----------------------------------------------------------------------------

class ImprovedQLearningAgent:
    """
    🧠 AGENTUL Q-LEARNING ÎMBUNĂTĂȚIT
    Versiune corectată cu sistem de recompense, experience replay și euristici.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.q_table = defaultdict(float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Statistici
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Pentru UI
        self.last_state = None
        self.last_action = None
        self.decision_reason = ""

        # Memory pentru experience replay
        self.memory = []
        self.max_memory = 1000

    def state_to_string(self, board):
        """Convertește tabla într-un string pentru Q-table"""
        return ''.join(c if c != '' else '_' for c in board)

    def get_valid_actions(self, board):
        """Găsește toate mișcările posibile"""
        return [i for i, cell in enumerate(board) if cell == '']

    def check_winner(self, board):
        """Verifică dacă cineva a câștigat"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for pattern in win_patterns:
            if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != '':
                return board[pattern[0]]
        return None

    def is_winning_move(self, board, position, player):
        """Verifică dacă o mutare câștigă"""
        if board[position] != '':
            return False
        test_board = board[:]
        test_board[position] = player
        return self.check_winner(test_board) == player

    def is_blocking_move(self, board, position, player):
        """Verifică dacă o mutare blochează adversarul"""
        opponent = 'X' if player == 'O' else 'O'
        return self.is_winning_move(board, position, opponent)

    def evaluate_position(self, board, action, player='O'):
        """Evaluează strategic o poziție"""
        score = 0.0
        if self.is_winning_move(board, action, player):
            score += 1.0
        if self.is_blocking_move(board, action, player):
            score += 0.8
        if action == 4:
            score += 0.3
        elif action in [0, 2, 6, 8]:
            score += 0.2
        return score * 0.1

    def choose_action(self, board, training=True):
        """Alege o acțiune folosind epsilon-greedy cu evaluare strategică"""
        state = self.state_to_string(board)
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return None

        # Verifică întâi mutările critice (câștig/blocare)
        for action in valid_actions:
            if self.is_winning_move(board, action, 'O'):
                self.decision_reason = f"🎯 CÂȘTIG: Poziția {action + 1}"
                self.last_state = state
                self.last_action = action
                return action

        for action in valid_actions:
            if self.is_blocking_move(board, action, 'O'):
                self.decision_reason = f"🛡️ BLOCARE: Poziția {action + 1}"
                self.last_state = state
                self.last_action = action
                return action

        # Epsilon-greedy pentru restul deciziilor
        epsilon = self.epsilon if training else max(0.05, self.epsilon * 0.1)
        if random.random() < epsilon and training:
            # Explorare cu preferință pentru mutări strategice
            weights = [1.0 + self.evaluate_position(board, action) for action in valid_actions]
            action = random.choices(valid_actions, weights=weights, k=1)[0]
            self.decision_reason = f"🎲 EXPLORARE: Poziția {action + 1} (ε={epsilon:.2f})"
        else:
            # Exploatare - folosește Q-values + euristică
            action_values = {a: self.q_table.get((state, a), 0.0) + self.evaluate_position(board, a) for a in
                             valid_actions}
            max_value = max(action_values.values())
            best_actions = [a for a, v in action_values.items() if abs(v - max_value) < 0.001]
            action = random.choice(best_actions)
            q_val = self.q_table.get((state, action), 0.0)
            self.decision_reason = f"🧠 Q-LEARNING: Poziția {action + 1} (Q={q_val:.3f})"

        self.last_state = state
        self.last_action = action
        return action

    # Explicatie 1
    def update_q_value(self, state, action, reward, next_state, done=False):
        """Actualizează Q-value folosind formula standard"""
        # Pasul 1: Citește scorul actual al acestei mutări din "memoria" AI-ului
        current_q = self.q_table.get((state, action), 0.0)

        # Pasul 2: Calculează scorul "țintă" (cât de bună ar trebui să fie mutarea)
        if done:
            # Dacă jocul s-a terminat, ținta este direct recompensa finală
            target = reward
        else:
            # Dacă jocul continuă, calculează valoarea mutării ținând cont de viitor
            next_board = [c if c != '_' else '' for c in next_state]
            # Găsește toate mutările posibile în următoarea stare
            next_valid_actions = self.get_valid_actions(next_board)
            # Alege cea mai bună mutare viitoare (cu cel mai mare scor)
            max_next_q = max((self.q_table.get((next_state, a), 0.0) for a in next_valid_actions), default=0.0)
            # Combină recompensa imediată cu valoarea viitoare (așa "vede AI-ul în viitor")
            target = reward + self.discount_factor * max_next_q

        # Pasul 3: Actualizează scorul învățând GRADUAL din diferența dintre așteptat și realitate
        # Formula: scor_nou = scor_vechi + rata_învățare * (țintă - scor_vechi)
        new_q = current_q + self.learning_rate * (target - current_q)

        # Pasul 4: Salvează noul scor în "memoria" AI-ului
        self.q_table[(state, action)] = new_q

        # Pasul 5: Adaugă această experiență în memorie pentru "revizuire" ulterioară
        self.memory.append((state, action, reward, next_state, done))
        # Dacă memoria e prea mare, șterge experiențele cele mai vechi
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)


    def experience_replay(self, batch_size=32):
        """Re-antrenează pe experiențe anterioare"""
        # Dacă nu avem suficiente experiențe salvate, nu face nimic
        if len(self.memory) < batch_size:
            return

        # Alege ALEATORIU 32 de experiențe din trecut (ca la revizuit pentru examen)
        batch = random.sample(self.memory, batch_size)

        # Re-învață din fiecare experiență aleasă
        for state, action, reward, next_state, done in batch:
            # Aplică din nou procesul de învățare pe experiența veche
            self.update_q_value(state, action, reward, next_state, done)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Reduce epsilon gradual"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def game_ended(self, reward):
        """Actualizează statisticile la final de joc (pt UI)"""
        self.total_games += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.last_state = None
        self.last_action = None

    def get_statistics(self):
        """Returnează statisticile"""
        if self.total_games == 0:
            return {"total": 0, "wins": 0, "losses": 0, "draws": 0, "win_rate": 0, "loss_rate": 0, "draw_rate": 0}
        return {
            "total": self.total_games, "wins": self.wins, "losses": self.losses, "draws": self.draws,
            "win_rate": self.wins / self.total_games, "loss_rate": self.losses / self.total_games,
            "draw_rate": self.draws / self.total_games
        }

    def save_model(self, filename):
        """Salvează modelul"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table), 'stats': self.get_statistics(),
                'params': {'learning_rate': self.learning_rate, 'discount_factor': self.discount_factor,
                           'epsilon': self.epsilon},
                'memory': self.memory[-100:] if self.memory else []
            }, f)

    def load_model(self, filename):
        """Încarcă un model salvat"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                stats = data.get('stats', {})
                self.total_games, self.wins, self.losses, self.draws = stats.get('total', 0), stats.get('wins',
                                                                                                        0), stats.get(
                    'losses', 0), stats.get('draws', 0)
                if 'params' in data:
                    self.learning_rate = data['params'].get('learning_rate', self.learning_rate)
                    self.discount_factor = data['params'].get('discount_factor', self.discount_factor)
                    self.epsilon = data['params'].get('epsilon', self.epsilon)
                if 'memory' in data:
                    self.memory = data['memory']
                return True
        except Exception as e:
            print(f"Eroare la încărcare: {e}")
            return False


class SmartOpponent:
    """Adversar inteligent pentru antrenament"""

    def __init__(self, difficulty=0.7):
        self.difficulty = difficulty
        self.agent = ImprovedQLearningAgent()  # Folosim logica de aici

    def choose_move(self, board):
        if random.random() < self.difficulty:
            # Mutare inteligentă
            valid_actions = self.agent.get_valid_actions(board)
            for p in ['X', 'O']:  # Câștigă dacă poate, blochează dacă trebuie
                for action in valid_actions:
                    if self.agent.is_winning_move(board, action, p):
                        return action
            if board[4] == '': return 4
            corners = [0, 2, 6, 8]
            random.shuffle(corners)
            for c in corners:
                if board[c] == '': return c
            return random.choice(valid_actions) if valid_actions else None
        else:
            # Mutare aleatorie
            valid_actions = self.agent.get_valid_actions(board)
            return random.choice(valid_actions) if valid_actions else None


class ImprovedTrainingThread(QThread):
    """Thread îmbunătățit pentru antrenament"""
    progress_updated = pyqtSignal(int, dict, str)
    training_completed = pyqtSignal()

    def __init__(self, agent, episodes=1000, training_mode='smart'):
        super().__init__()
        self.agent = agent
        self.episodes = episodes
        self.training_mode = training_mode
        self.should_stop = False
        self.opponents = {
            'random': SmartOpponent(0.0), 'easy': SmartOpponent(0.3),
            'medium': SmartOpponent(0.6), 'hard': SmartOpponent(0.9),
            'expert': SmartOpponent(1.0), 'curriculum': [SmartOpponent(0.1 * i) for i in range(11)]
        }

    # Explicatie 1
    def play_game(self, opponent):
        # Creează o tablă goală
        board = [''] * 9

        # Lista care ține minte toate mutările din acest joc
        history = []
        # Alege aleatoriu cine începe (X sau O)
        turn = random.choice(['X', 'O'])

        while True:
            # Verifică dacă cineva a câștigat
            winner = self.agent.check_winner(board)

            # Dacă jocul s-a terminat (cineva a câștigat SAU tabla e plină)
            if winner or '' not in board:
                # Mapare: O câștigă = +1, X câștigă = -1
                reward_map = {'O': 1.0, 'X': -1.0}
                # Recompensa finală: +1 dacă AI (O) câștigă, -1 dacă pierde, +0.1 dacă egal
                final_reward = reward_map.get(winner, 0.1)

                # Parcurge TOATE mutările din joc DE LA FINAL LA ÎNCEPUT
                for state, action, next_state in reversed(history):
                    # Fiecare mutare învață din rezultatul final
                    self.agent.update_q_value(state, action, final_reward, next_state, done=True)
                    # Mutările de la început primesc mai puțină vină/laudă decât cele de la final
                    # Exemplu: ultima mutare primește 1.0, penultima -0.5, ante-penultima 0.25, etc.
                    final_reward *= -0.5

                # Actualizează statisticile pentru UI
                self.agent.game_ended(1.0 if winner == 'O' else -1.0 if winner == 'X' else 0.0)
                break  # Termină jocul

            # Dacă e rândul AI-ului (O)
            if turn == 'O':
                # Salvează starea curentă a tablei
                state = self.agent.state_to_string(board)
                # AI-ul alege o mutare (cu explorare/exploatare)
                action = self.agent.choose_action(board, training=True)
                if action is not None:
                    # Pune O pe tablă
                    board[action] = 'O'
                    # Salvează noua stare a tablei
                    next_state = self.agent.state_to_string(board)
                    # Adaugă mutarea în istoric pentru a învăța la final
                    history.append((state, action, next_state))
                # Schimbă rândul la adversar
                turn = 'X'
            else:
                # Adversarul (X) face mutarea sa
                move = opponent.choose_move(board)
                if move is not None:
                    board[move] = 'X'
                # Schimbă rândul înapoi la AI
                turn = 'O'

    def run(self):
        if self.training_mode == 'curriculum':
            levels = self.opponents['curriculum']
            episodes_per_level = self.episodes // len(levels)
            for i, opponent in enumerate(levels):
                if self.should_stop: break
                for episode in range(episodes_per_level):
                    if self.should_stop: break
                    self.play_game(opponent)
                    if episode % 10 == 0: self.agent.experience_replay(); self.agent.decay_epsilon()
                    if episode % 50 == 0:
                        progress = ((i * episodes_per_level) + episode) * 100 // self.episodes
                        status = f"Curriculum Nivel {i + 1}/{len(levels)} (Dificultate: {opponent.difficulty:.1f})"
                        self.progress_updated.emit(progress, self.agent.get_statistics(), status)
                        time.sleep(0.01)
        else:
            opponent = self.opponents.get(self.training_mode, self.opponents['medium'])
            for episode in range(self.episodes):
                if self.should_stop: break
                self.play_game(opponent)
                if episode % 10 == 0: self.agent.experience_replay(); self.agent.decay_epsilon()
                if episode % 50 == 0:
                    progress = episode * 100 // self.episodes
                    status = f"Antrenament vs '{self.training_mode}'"
                    self.progress_updated.emit(progress, self.agent.get_statistics(), status)
                    time.sleep(0.01)
        self.training_completed.emit()

    def stop_training(self):
        self.should_stop = True


# -----------------------------------------------------------------------------
# CLASA PRINCIPALĂ PENTRU INTERFAȚA GRAFICĂ
# -----------------------------------------------------------------------------

class TicTacToeAI(QWidget):
    """🎮 Interfața principală - versiune integrată"""

    def __init__(self):
        super().__init__()
        self.ml_agent = ImprovedQLearningAgent()
        self.training_thread = None
        if self.ml_agent.load_model('xo_ai_model_improved.pkl'):
            print("✅ Model îmbunătățit încărcat!")
        elif self.ml_agent.load_model('xo_ai_model.pkl'):
            print("⚠️ Model vechi încărcat - consideră re-antrenarea!")
        else:
            print("ℹ️ Model nou creat - începe antrenamentul!")
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🧠 X și O - Machine Learning (Versiune Îmbunătățită)')
        self.setGeometry(50, 50, 1000, 700)
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        game_widget = self.create_game_panel()
        training_widget = self.create_improved_training_panel()
        splitter.addWidget(game_widget)
        splitter.addWidget(training_widget)
        splitter.setSizes([450, 550])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.reset_game()

    def create_game_panel(self):
        widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel('🎯 Joacă împotriva AI-ului Antrenat!')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title)
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        self.thinking_label = QLabel()
        self.thinking_label.setAlignment(Qt.AlignCenter)
        self.thinking_label.setStyleSheet(
            "font-size: 14px; background-color: #f39c12; color: white; padding: 8px; border-radius: 5px; font-weight: bold;")
        self.thinking_label.hide()
        layout.addWidget(self.thinking_label)
        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(10)
        self.buttons = []
        for i in range(9):
            button = QPushButton('')
            button.setFixedSize(100, 100)
            index = i
            button.clicked.connect(lambda _, idx=index: self.on_button_click(idx))
            self.buttons.append(button)
            grid_layout.addWidget(button, i // 3, i % 3)
        layout.addWidget(grid_container, 0, Qt.AlignCenter)
        self.new_game_btn = QPushButton('🔄 Joc Nou')
        self.new_game_btn.clicked.connect(self.reset_game)
        self.new_game_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.new_game_btn, 0, Qt.AlignCenter)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_improved_training_panel(self):
        widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel('🎓 Antrenament Avansat')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        self.stats_label = QLabel(self.get_stats_text())
        self.stats_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; font-family: 'Courier New', monospace;")
        layout.addWidget(self.stats_label)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mod antrenament:"))
        self.training_mode = QComboBox()
        self.training_mode.addItems(['curriculum', 'random', 'easy', 'medium', 'hard', 'expert'])
        self.training_mode.setToolTip(
            "• curriculum: Progresie graduală (recomandat!)\n• random: Adversar complet aleatoriu\n• easy/medium/hard: Dificultate fixă\n• expert: Adversar perfect")
        mode_layout.addWidget(self.training_mode)
        layout.addLayout(mode_layout)
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("Episoade:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(100, 100000)
        self.episodes_spin.setValue(5000)
        self.episodes_spin.setSingleStep(100)
        episodes_layout.addWidget(self.episodes_spin)
        layout.addLayout(episodes_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.training_status = QLabel('')
        self.training_status.setAlignment(Qt.AlignCenter)
        self.training_status.setStyleSheet("color: #e67e22; font-weight: bold;")
        layout.addWidget(self.training_status)
        training_btns = QHBoxLayout()
        self.train_btn = QPushButton('🏃 Antrenează')
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        training_btns.addWidget(self.train_btn)
        self.stop_btn = QPushButton('⏹️ Stop')
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        training_btns.addWidget(self.stop_btn)
        layout.addLayout(training_btns)
        model_btns = QHBoxLayout()
        save_btn = QPushButton('💾 Salvează')
        save_btn.clicked.connect(self.save_model)
        model_btns.addWidget(save_btn)
        load_btn = QPushButton('📁 Încarcă')
        load_btn.clicked.connect(self.load_model)
        model_btns.addWidget(load_btn)
        layout.addLayout(model_btns)
        info = QLabel("📊 PARAMETRI: Learning rate: 0.1, Discount: 0.95, Epsilon: variabil\n"
                      "🎯 SFATURI:\n"
                      "1. Folosește 'curriculum' pentru cel mai bun rezultat.\n"
                      "2. 5000+ episoade pentru performanță bună.\n"
                      "3. Salvează după fiecare sesiune de antrenament.")
        info.setStyleSheet("""
            background-color: #e8f5e9; padding: 10px; border-radius: 5px;
            font-size: 11px; border-left: 4px solid #4caf50;""")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def get_stats_text(self):
        stats = self.ml_agent.get_statistics()
        epsilon_text = f"{self.ml_agent.epsilon:.3f}" if hasattr(self.ml_agent, 'epsilon') else "N/A"
        return f"""
📊 STATISTICI MODEL:
━━━━━━━━━━━━━━━━━━
Jocuri totale: {stats['total']:,}
Victorii: {stats['wins']:,} ({stats['win_rate']:.1%})
Înfrângeri: {stats['losses']:,} ({stats['loss_rate']:.1%})
Egalități: {stats['draws']:,} ({stats['draw_rate']:.1%})

Q-Table: {len(self.ml_agent.q_table):,} intrări
Epsilon: {epsilon_text}
Memorie: {len(self.ml_agent.memory)} experiențe"""

    def reset_game(self):
        self.board = [''] * 9
        self.player_turn = True
        self.game_over = False
        self.last_ai_move = None
        for button in self.buttons:
            button.setText('')
            button.setEnabled(True)
            button.setStyleSheet("""
                QPushButton {font-size: 40px; font-weight: bold; border: 2px solid #34495e; border-radius: 8px; background-color: #ecf0f1;}
                QPushButton:hover {background-color: #bdc3c7;}""")
        self.status_label.setText('Rândul tău (X)')
        self.status_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #3498db; color: white; padding: 10px; border-radius: 5px;")
        self.thinking_label.hide()

    def on_button_click(self, index):
        if not self.player_turn or self.game_over or self.board[index] != '': return
        self.board[index] = 'X'
        self.buttons[index].setText('X')
        self.buttons[index].setStyleSheet(
            "font-size: 40px; font-weight: bold; border: 2px solid #34495e; border-radius: 8px; background-color: #3498db; color: white;")
        if self.check_game_end(): return
        self.player_turn = False
        self.status_label.setText('🤖 AI-ul se gândește...')
        self.thinking_label.setText('🧠 Analizez pozițiile...')
        self.thinking_label.show()
        QTimer.singleShot(800, self.ai_move)

    def ai_move(self):
        action = self.ml_agent.choose_action(self.board, training=False)
        if action is not None:
            self.board[action] = 'O'
            self.buttons[action].setText('O')
            self.buttons[action].setStyleSheet(
                "font-size: 40px; font-weight: bold; border: 2px solid #34495e; border-radius: 8px; background-color: #e74c3c; color: white;")
            self.last_ai_move = action
            self.thinking_label.setText(self.ml_agent.decision_reason)
            QTimer.singleShot(1500, self.thinking_label.hide)
        if not self.check_game_end():
            self.player_turn = True
            self.status_label.setText('Rândul tău (X)')

    def check_game_end(self):
        winner = self.check_winner()
        if winner:
            self.game_over = True
            self.thinking_label.hide()
            if winner == 'X':
                self.status_label.setText('🎉 Ai câștigat!')
                self.status_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; background-color: #27ae60; color: white; padding: 10px; border-radius: 5px;")
            else:
                self.status_label.setText('🤖 AI-ul a câștigat!')
                self.status_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px;")
            for button in self.buttons: button.setEnabled(False)
            self.stats_label.setText(self.get_stats_text())
            return True
        elif '' not in self.board:
            self.game_over = True
            self.status_label.setText('🤝 Egalitate!')
            self.status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; background-color: #f39c12; color: white; padding: 10px; border-radius: 5px;")
            self.thinking_label.hide()
            for button in self.buttons: button.setEnabled(False)
            self.stats_label.setText(self.get_stats_text())
            return True
        return False

    def check_winner(self):
        win_conditions = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]] != '':
                for idx in cond: self.buttons[idx].setStyleSheet(
                    self.buttons[idx].styleSheet() + "border: 4px solid #f1c40f;")
                return self.board[cond[0]]
        return None

    def start_training(self):
        self.training_thread = ImprovedTrainingThread(self.ml_agent, self.episodes_spin.value(),
                                                      self.training_mode.currentText())
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_completed.connect(self.training_finished)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.training_status.setText(f"Antrenament {self.training_mode.currentText()}...")
        self.training_thread.start()

    def update_training_progress(self, progress, stats, status):
        self.progress_bar.setValue(progress)
        self.stats_label.setText(self.get_stats_text())
        self.training_status.setText(status)

    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.training_status.setText("✅ Antrenament complet!")
        self.ml_agent.save_model('xo_ai_model_improved.pkl')
        QTimer.singleShot(3000, lambda: self.training_status.setText(""))

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop_training()
            self.training_status.setText("⏹️ Antrenament oprit...")

    def save_model(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Salvează Model", "xo_ai_model_improved.pkl",
                                                  "Pickle Files (*.pkl)")
        if filename:
            self.ml_agent.save_model(filename)
            self.training_status.setText("💾 Model salvat!")
            QTimer.singleShot(2000, lambda: self.training_status.setText(""))

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Încarcă Model", "", "Pickle Files (*.pkl)")
        if filename and self.ml_agent.load_model(filename):
            self.stats_label.setText(self.get_stats_text())
            self.training_status.setText("📁 Model încărcat!")
            QTimer.singleShot(2000, lambda: self.training_status.setText(""))
        elif filename:
            self.training_status.setText("❌ Eroare la încărcare!")


# -----------------------------------------------------------------------------
# BLOCUL DE EXECUȚIE PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     🧠 X și O cu Machine Learning - Versiune Îmbunătățită     ║
    ╚════════════════════════════════════════════════════════════════╝

    🎯 CARACTERISTICI NOI:
    ━━━━━━━━━━━━━━━━━━━━
    ✅ Sistem de recompense îmbunătățit
    ✅ Antrenament cu adversari inteligenți (Curriculum Learning)
    ✅ Experience Replay pentru stabilitate
    ✅ Epsilon Decay automat
    ✅ Compatibil cu modele vechi și noi

    📊 MODURI DE ANTRENAMENT:
    ━━━━━━━━━━━━━━━━━━━━━━━
    • CURRICULUM: Cel mai bun! Progresie de la ușor la expert.
    • RANDOM/EASY/MEDIUM/HARD/EXPERT: Dificultate fixă.

    💡 RECOMANDĂRI:
    ━━━━━━━━━━━━━━━
    1. Folosește modul CURRICULUM pentru 5000+ episoade.
    2. Salvează după fiecare sesiune de antrenament.
    3. Urmărește rata de victorii - ținta este >70% vs 'medium'.

    Mult succes! 🎮
    """)
    app = QApplication(sys.argv)
    game = TicTacToeAI()
    game.show()
    sys.exit(app.exec_())




