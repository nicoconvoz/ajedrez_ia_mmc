# -*- coding: utf-8 -*-
"""
MMC Chess Learner – v47  (23‑may‑2025)
-----------------------------------------------------------
* Usa directamente la base MMC existente; no re‑importa PGN.
* Confía en la memoria en todas las jugadas (85 % si existe arista).
* MCTS más profundo (hasta 300 simulaciones).
* Resto de lógica igual a v45.
-----------------------------------------------------------
* FIX: se corrigen todos los errores de indentación y los métodos que
  habían quedado anidados por error en v43.
* Se eliminan duplicados (choose/evaluate/ChessGUI) y la función evaluate()
  vuelve a ser un método de Agent.
* Se mantiene la lógica añadida en v43 para rellenar la tabla *features*.
"""

from __future__ import annotations
import threading, queue, time, random, json, os, math, hashlib, collections
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import mysql.connector
import chess, chess.pgn

from PIL import Image, ImageTk, ImageDraw
from PIL.Image import Resampling

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities

# ---------------------------------------------------------------------------
# Parámetros globales
# ---------------------------------------------------------------------------
IMAGES_DIR = "images"
SCORE_FILE = "scoreboard.json"
MACHINE_NAME = "_machine_"
THINK_MS = 40

REP_PENALTY = 0.75        # castigo en evaluate cuando hay repetición
DRAW_PENALTY = -10.0      # valor devuelto cuando se disuade un empate
MAX_PLIES   = 400         # tablas forzadas luego de 400 medios‑movimientos

sha = lambda t: hashlib.sha256(t.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def canon(board: chess.Board) -> str:
    """Devuelve las 4 primeras partes del FEN como clave estable"""
    return " ".join(board.fen().split()[:4])


def elo_exp(ra, rb):               # expectativa de victoria de ra sobre rb
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def elo_update(ra, rb, sa, k=32):  # nuevo elo tras el resultado sa (1,½,0)
    return ra + k * (sa - elo_exp(ra, rb)), rb + k * ((1 - sa) - elo_exp(rb, ra))


def load_scores():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_scores(s):
    with open(SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


def reward(prev: chess.Board, new: chess.Board, mv: chess.Move) -> float:
    """Pequeña función de recompensa para refuerzo de aristas"""
    r = 0.10
    if prev.is_capture(mv):
        r += 0.5
    if prev.is_check() and not new.is_check():
        r += 0.4
    if new.is_check():
        r += 0.2
    if new.is_checkmate():
        r += 1
    return r


# ---------------------------------------------------------------------------
# Extracción de características de posición
# ---------------------------------------------------------------------------
def _extract_features(board: chess.Board) -> dict[str, int]:
    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    mat = sum(
        piece_values[p.piece_type] if p.color else -piece_values[p.piece_type]
        for p in board.piece_map().values()
    )
    attackers_w = defenders_w = attackers_b = defenders_b = 0
    for sq, pc in board.piece_map().items():
        attacked = board.attackers(not pc.color, sq)
        defended = board.attackers(pc.color, sq)
        if pc.color:
            attackers_w += bool(attacked)
            defenders_w += bool(defended)
        else:
            attackers_b += bool(attacked)
            defenders_b += bool(defended)
    ks_w = int(board.is_check() and board.turn)
    ks_b = int(board.is_check() and not board.turn)
    mob_w = len(list(board.legal_moves)) if board.turn else 0
    mob_b = len(list(board.legal_moves)) if not board.turn else 0
    return dict(
        mat_diff=mat,
        attackers_w=attackers_w,
        attackers_b=attackers_b,
        defenders_w=defenders_w,
        defenders_b=defenders_b,
        king_safety_w=ks_w,
        king_safety_b=ks_b,
        mobility_w=mob_w,
        mobility_b=mob_b,
    )


# ---------------------------------------------------------------------------
# Base de datos MMC
# ---------------------------------------------------------------------------
class MMC:
    """Memoria de Mapa Conceptual para posiciones de ajedrez."""

    def __init__(self, cfg: str = "db_config.json"):
        self.cfg = json.load(open(cfg, "r", encoding="utf-8"))
        db_name = self.cfg.get("database", "mmc_chess")
        cfg_nodb = {k: v for k, v in self.cfg.items() if k != "database"}

        # asegurar BD existente
        tmp = mysql.connector.connect(**cfg_nodb, autocommit=True)
        cur = tmp.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cur.close()
        tmp.close()

        # conexión principal + hilo escritor
        self.conn = mysql.connector.connect(**self.cfg, autocommit=True)
        self.write_q: queue.Queue[tuple[str, tuple]] = queue.Queue()
        threading.Thread(target=self._flush_worker, daemon=True).start()

        # hiper‑parámetros comportamiento
        self.exploration, self.lateral_eps = 0.15, 0.20
        self.decay_rate, self.base_prune = 0.0002, 0.50
        self.last_fens = collections.deque(maxlen=32)

        self._schema()

    # ------------------------------------------------------------------
    # Esquema de tablas
    # ------------------------------------------------------------------
    def _schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS nodes(
                fen VARCHAR(120) PRIMARY KEY,
                last_update DOUBLE)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS edges(
                src VARCHAR(120), dst VARCHAR(120),
                weight DOUBLE DEFAULT 0,
                capture TINYINT DEFAULT 0,
                `escape` TINYINT DEFAULT 0,
                `check`  TINYINT DEFAULT 0,
                mate      TINYINT DEFAULT 0,
                draw_push TINYINT DEFAULT 0,
                win INT DEFAULT 0,
                loss INT DEFAULT 0,
                PRIMARY KEY(src,dst),
                INDEX(src), INDEX(dst))"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS features(
                fen VARCHAR(120) PRIMARY KEY,
                mat_diff SMALLINT,
                attackers_w TINYINT,
                attackers_b TINYINT,
                defenders_w TINYINT,
                defenders_b TINYINT,
                king_safety_w TINYINT,
                king_safety_b TINYINT,
                mobility_w TINYINT,
                mobility_b TINYINT,
                last_update DOUBLE)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS games(
                id INT AUTO_INCREMENT PRIMARY KEY,
                pgn MEDIUMTEXT,
                result VARCHAR(7),
                player VARCHAR(64),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
        # migraciones
        cur.execute("SHOW COLUMNS FROM edges")
        cols = {r[0] for r in cur.fetchall()}
        needed = {
            "win": "INT DEFAULT 0",
            "loss": "INT DEFAULT 0",
            "check": "TINYINT DEFAULT 0",
            "mate": "TINYINT DEFAULT 0",
            "draw_push": "TINYINT DEFAULT 0",
            "escape": "TINYINT DEFAULT 0",
        }
        mig = [f"ADD COLUMN `{c}` {defn}" for c, defn in needed.items() if c not in cols]
        if mig:
            cur.execute("ALTER TABLE edges " + ", ".join(mig))
        cur.close()

    # ------------------------------------------------------------------
    # Conectores
    # ------------------------------------------------------------------
    def _get_conn(self):
        """Conexión por hilo (pool barato)"""
        if not hasattr(self, "_local"):
            self._local = threading.local()
        cn = getattr(self._local, "conn", None)
        try:
            alive = cn is not None and cn.is_connected()
        except Exception:
            alive = False
        if not alive:
            cfg = dict(self.cfg)
            cfg.setdefault("database", "mmc_chess")
            cn = mysql.connector.connect(**cfg, autocommit=True)
            self._local.conn = cn
        return cn

    def q(self, sql: str, params: tuple = ()):
        cur = self._get_conn().cursor()
        cur.execute(sql, params)
        return cur

    # ------------------------------------------------------------------
    # Escritor asíncrono
    # ------------------------------------------------------------------
    def enqueue(self, sql: str, params: tuple):
        self.write_q.put((sql, params))

    def _flush_worker(self):
        while True:
            batch = []
            try:
                batch.append(self.write_q.get(timeout=0.5))
            except queue.Empty:
                continue
            while len(batch) < 128:
                try:
                    batch.append(self.write_q.get_nowait())
                except queue.Empty:
                    break
            cfg = dict(self.cfg); cfg.setdefault("database", "mmc_chess")
            conn = mysql.connector.connect(**cfg, autocommit=True)
            cur = conn.cursor()
            for sql, params in batch:
                cur.execute(sql, params)
                self.write_q.task_done()
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Características de posición
    # ------------------------------------------------------------------
    def _store_features(self, fen: str):
        feats = _extract_features(chess.Board(fen))
        now = time.time()
        self.enqueue(
            """INSERT INTO features(
                    fen, mat_diff, attackers_w, attackers_b, defenders_w, defenders_b,
                    king_safety_w, king_safety_b, mobility_w, mobility_b, last_update)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    mat_diff=VALUES(mat_diff),
                    attackers_w=VALUES(attackers_w),
                    attackers_b=VALUES(attackers_b),
                    defenders_w=VALUES(defenders_w),
                    defenders_b=VALUES(defenders_b),
                    king_safety_w=VALUES(king_safety_w),
                    king_safety_b=VALUES(king_safety_b),
                    mobility_w=VALUES(mobility_w),
                    mobility_b=VALUES(mobility_b),
                    last_update=VALUES(last_update)""",
            (
                fen,
                feats["mat_diff"],
                feats["attackers_w"],
                feats["attackers_b"],
                feats["defenders_w"],
                feats["defenders_b"],
                feats["king_safety_w"],
                feats["king_safety_b"],
                feats["mobility_w"],
                feats["mobility_b"],
                now,
            ),
        )

    # ------------------------------------------------------------------
    # Núcleo de actualización
    # ------------------------------------------------------------------
    def update(
        self,
        src_fen: str,
        dst_fen: str,
        delta: float,
        capture: int = 0,
        escape_: int = 0,
        check: int = 0,
        mate: int = 0,
        draw_push: int = 0,
        win: int = 0,
        loss: int = 0,
    ):
        """Refuerza la arista (src→dst) y almacena features."""
        now = time.time()

        # nodos
        self.enqueue(
            "INSERT INTO nodes VALUES(%s,%s) ON DUPLICATE KEY UPDATE last_update=%s",
            (src_fen, now, now),
        )
        self.enqueue(
            "INSERT INTO nodes VALUES(%s,%s) ON DUPLICATE KEY UPDATE last_update=%s",
            (dst_fen, now, now),
        )

        # características
        self._store_features(src_fen)
        self._store_features(dst_fen)

        # arista
        self.enqueue(
            """INSERT INTO edges(src,dst,weight,win,loss,capture,`escape`,`check`,mate,draw_push)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    weight       = weight + %s,
                    win          = win    + %s,
                    loss         = loss   + %s,
                    capture      = capture | %s,
                    `escape`     = `escape`| %s,
                    `check`      = `check` | %s,
                    mate         = mate    | %s,
                    draw_push    = draw_push | %s""",
            (
                src_fen,
                dst_fen,
                delta,
                win,
                loss,
                capture,
                escape_,
                check,
                mate,
                draw_push,
                delta,
                win,
                loss,
                capture,
                escape_,
                check,
                mate,
                draw_push,
            ),
        )

    # ------------------------------------------------------------------
    # Métodos auxiliares públicos
    # ------------------------------------------------------------------
    def get_edge(self, src: str, dst: str):
        return self.q(
            "SELECT src,dst,weight,win,loss,capture,`escape` FROM edges "
            "WHERE src=%s AND dst=%s",
            (src, dst),
        ).fetchone()

    def get_features(self, fen: str):
        return self.q(
            """SELECT mat_diff, attackers_w, attackers_b, defenders_w, defenders_b,
                      king_safety_w, king_safety_b, mobility_w, mobility_b
               FROM features WHERE fen=%s""",
            (fen,),
        ).fetchone()

    
def choose(self, fen: str, creative: bool = False):
    """
    Elige un destino para la posición `fen` consultando la memoria MMC.

    * Usa como puntaje base el peso `weight`.
    * Refuerza capturas y penaliza las aristas marcadas como escapatoria.
    * Pondera la tasa de victoria histórica (win / (win+loss)).
    * Siempre que exista al menos una arista con puntaje claramente mayor,
      la IA será más “determinista”; de lo contrario explorará.
    """

    rows = self.q(
        "SELECT dst, weight, capture, `escape`, win, loss FROM edges WHERE src=%s",
        (fen,),
    ).fetchall()
    if not rows:
        return None

    # calcular puntuación compuesta
    scores = []
    for dst, w, cap, esc, win, loss in rows:
        wl_ratio = (win + 1) / (win + loss + 2)        # suavizado
        score = (w + 1e-3) * (1 + 0.75 * cap - 0.40 * esc) * (0.5 + wl_ratio)
        scores.append(score)

    # si hay una clara mejor (20 % por encima), tomarla salvo que creative
    best_idx = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_idx]
    avg = sum(scores) / len(scores)

    if not creative and best_score > 1.2 * avg and random.random() > self.exploration:
        return rows[best_idx][0]

    # exploración ponderada
    return random.choices(
        [r[0] for r in rows],
        weights=scores,
        k=1
    )[0]

    # --- mantenimiento --- --- mantenimiento ---
    def decay(self):
        f = math.exp(-self.decay_rate * 60)
        self.q("UPDATE edges SET weight = weight * %s WHERE win = 0", (f,))
        self.q(
            "DELETE FROM edges WHERE weight < %s AND win = 0 AND loss > 5",
            (self.base_prune,),
        )

    # --- snapshot para visualizaciones ---
    def snapshot(self):
        G = nx.Graph()
        G.add_edges_from(self.q("SELECT src,dst FROM edges LIMIT 3000").fetchall())
        comms = louvain_communities(G, seed=42) if G.number_of_nodes() else []
        return G, {i: list(c) for i, c in enumerate(comms)}

    # --- almacenamiento de partidas ---
    def store_game(self, pgn: str, res: str, player: str):
        self.enqueue(
            "INSERT INTO games(pgn,result,player) VALUES(%s,%s,%s)",
            (pgn, res, player),
        )

    def import_pgn(self, path: str, player: str = "import") -> int:
        """Importa partidas de un archivo PGN reforzando la MMC."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        n_games = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                self.store_game(str(game), result, player)

                board = game.board()
                for mv in game.mainline_moves():
                    from_fen = canon(board)
                    capture = 1 if board.is_capture(mv) else 0
                    escape_before = board.is_check()
                    board.push(mv)
                    to_fen = canon(board)

                    winner_is_white = result == "1-0"
                    color_to_move = not board.turn  # posición `from_fen` es del color opuesto
                    win = int((color_to_move and winner_is_white) or (not color_to_move and not winner_is_white))
                    loss = int((color_to_move and not winner_is_white) or (not color_to_move and winner_is_white))

                    self.update(
                        from_fen,
                        to_fen,
                        0.05,
                        capture=capture,
                        escape_=int(escape_before),
                        win=win,
                        loss=loss,
                    )
                n_games += 1
        return n_games

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Agente IA
# ---------------------------------------------------------------------------
class Agent:
    MAT = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

    PST = {
        'P': [0, 5, 5, -5, -5, 5, 10, 0] * 8,
        'N': [-50, -40, -30, -30, -30, -30, -40, -50] * 8,
        'B': [-20, -10, -10, -10, -10, -10, -10, -20] * 8,
        'R': [0, 0, 0, 5, 5, 0, 0, 0] * 8,
        'Q': [-10, -5, -5, -5, -5, -5, -5, -10] * 8,
        'K': [-30, -40, -40, -50, -50, -40, -40, -30] * 8
    }

    def __init__(self, mmc: MMC):
        self.mmc = mmc
        self.prev_fen: str | None = None
        self.prev_board: chess.Board | None = None
        self.path: list[tuple[str, str, int, int]] = []  # src,dst,capture,escape

    # ------------------------------------------------------
    # Registro de movimientos en la MMC
    # ------------------------------------------------------
    def record(self, board: chess.Board, mv: chess.Move | None = None):
        fen = canon(board)
        if self.prev_fen:
            cap = int(self.prev_board.is_capture(mv)) if mv else 0
            esc = int(self.prev_board.is_check() and not board.is_check()) if mv else 0
            self.mmc.update(
                self.prev_fen,
                fen,
                reward(self.prev_board, board, mv) if mv else 0.05,
                cap,
                esc,
            )
            self.path.append((self.prev_fen, fen, cap, esc))
        self.prev_fen, self.prev_board = fen, board.copy(stack=False)
        self.mmc.last_fens.append(fen)

    
def finalize(self, win_flag: int, loss_flag: int):
    """
    Refuerza toda la trayectoria de la partida con un delta fuerte
    dependiendo del resultado.  Se refuerza  +1 por victoria,
    –1 por derrota, 0 por tablas.
    """
    delta = 1.0 if win_flag else -1.0 if loss_flag else 0.0
    for a, b, cap, esc in self.path:
        self.mmc.update(a, b, delta, cap, esc, win_flag, loss_flag)
    self.path.clear()

    # ------------------------------------------------------
    # Elección de movimiento
    # ------------------------------------------------------
    
def select(self, board: chess.Board) -> chess.Move:
    """
    Selecciona un movimiento para `board`:
    1. Intenta usar la memoria MMC de manera determinista, si existe
       una arista dominante (>60 % del total de puntuación).
    2. Si no hay un movimiento claramente ganador, mezcla una elección
       ponderada por la memoria con una búsqueda MCTS más profunda.
    """
    fen = canon(board)

    # ---------- 1) MEMORIA ----------
    tgt = self.mmc.choose(fen)
    if tgt:
        # ver cuán dominante es
        rows = self.mmc.q(
            "SELECT dst, weight FROM edges WHERE src=%s", (fen,)
        ).fetchall()
        total = sum(w for _, w in rows) or 1e-6
        dom = next((w for d, w in rows if d == tgt), 0) / total
        if dom > 0.60 or random.random() > self.mmc.exploration:
            for mv in board.legal_moves:
                b2 = board.copy(); b2.push(mv)
                if canon(b2) == tgt:
                    return mv

    # ---------- 2) MCTS ----------
    sims = min(500, 80 + len(list(board.legal_moves)) * 4)
    return self._mcts(board, sims=sims)
def _mcts(self, board: chess.Board, sims: int = 120) -> chess.Move:
        root_fen = canon(board)
        children: dict[str, list[tuple[chess.Move, str, float]]] = {root_fen: []}
        N = collections.Counter()  # visitas por arista
        W = collections.Counter()  # valor acumulado
        P = {}                     # prior

        def expand(fen: str):
            # asegurar clave y evitar KeyError
            if fen not in children:
                children[fen] = []
            if children[fen]:
                return
            if children[fen]:
                return
            moves = list(chess.Board(fen).legal_moves)
            priors = []
            for mv in moves:
                b2 = chess.Board(fen); b2.push(mv)
                tgt = canon(b2)
                p = 1.0
                row = self.mmc.get_edge(fen, tgt)
                if row:
                    _, _, w, win, loss, cap, esc = row
                    p = w + 0.5 * win - 0.3 * loss + 0.1 * cap - 0.1 * esc + 1e-3
                priors.append((mv, tgt, p))
            s = sum(p for _, _, p in priors)
            if not priors:
                children[fen] = []
                return
            s = max(1e-6, s)
            priors = [(mv, tgt, p / s) for mv, tgt, p in priors]
            children[fen] = priors
            for _, tgt, p in priors:
                P[(fen, tgt)] = p

        def simulate():
            path = []
            fen = root_fen
            expand(fen)
            while True:
                # selección UCT
                best_mv, best_tgt, best_u = None, None, -1
                for mv, tgt, _ in children[fen]:
                    u = W[(fen, tgt)] / (1 + N[(fen, tgt)]) + \
                        1.4 * P[(fen, tgt)] * math.sqrt(N[fen] + 1) / (1 + N[(fen, tgt)])
                    if u > best_u:
                        best_u, best_mv, best_tgt = u, mv, tgt
                path.append((fen, best_tgt))
                fen = best_tgt
                if fen not in children:
                    expand(fen)
                if chess.Board(fen).is_game_over() or len(path) > 4:
                    break
            val = self.evaluate(chess.Board(fen))
            for s, a in path:
                N[(s, a)] += 1
                W[(s, a)] += val

        for _ in range(sims):
            simulate()

        # mejor movimiento = más visitado
        best_mv, best_visits = None, -1
        for mv, tgt, _ in children[root_fen]:
            if N[(root_fen, tgt)] > best_visits:
                best_visits = N[(root_fen, tgt)]
                best_mv = mv

        return best_mv or random.choice(list(board.legal_moves))

    # ------------------------------------------------------
    # Evaluación estática
    # ------------------------------------------------------
def evaluate(self, board: chess.Board) -> float:
    if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return DRAW_PENALTY

        row = self.mmc.get_features(canon(board))
        if row:
            mat, at_w, at_b, de_w, de_b, ks_w, ks_b, mob_w, mob_b = row
            sign = 1 if board.turn else -1
            mat  = sign * mat
            atk  = (at_w - at_b) if board.turn else (at_b - at_w)
            dfn  = (de_w - de_b) if board.turn else (de_b - de_w)
            ks   = (ks_b - ks_w) if board.turn else (ks_w - ks_b)
            mob  = (mob_w - mob_b) if board.turn else (mob_b - mob_w)
            return mat + 0.05 * atk - 0.03 * dfn + 0.4 * mob + 0.4 * ks

            # fallback: material simple + check
        val = 0.0
        for p in board.piece_map().values():
            val += self.MAT[p.piece_type] * (1 if p.color == board.turn else -1)
        if board.is_check():
            val += 0.25
        return val


# ---------------------------------------------------------------------------
# Interfaz gráfica
# ---------------------------------------------------------------------------
class ChessGUI:
    def __init__(self, cfg="db_config.json"):
        self.scores = load_scores()
        self.scores.setdefault(MACHINE_NAME, {"pass": "", "elo": 1500})

        self.player = self._login()
        if not self.player:
            return

        self.mmc = MMC(cfg)
        self.agent = Agent(self.mmc)
        self.queue: queue.Queue[chess.Move] = queue.Queue()

        self._welcome_screen()

    # ------------------------------------------------------------------
    # Pantalla de login
    # ------------------------------------------------------------------
    def _login(self) -> str | None:
        root = tk.Tk()
        root.title("Inicio de sesión – MMC Chess")
        root.geometry("340x200")
        root.resizable(False, False)

        tk.Label(root, text="MMC Chess", font=("Arial", 16, "bold")).pack(pady=5)
        frm = tk.Frame(root); frm.pack(pady=5)

        tk.Label(frm, text="Usuario:").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        tk.Label(frm, text="Contraseña:").grid(row=1, column=0, sticky="e", padx=5, pady=4)

        user_entry = tk.Entry(frm, width=22)
        pass_entry = tk.Entry(frm, width=22, show="*")
        user_entry.grid(row=0, column=1, padx=5, pady=4)
        pass_entry.grid(row=1, column=1, padx=5, pady=4)
        info = tk.Label(root, text="", fg="red"); info.pack()
        result: list[str] = []

        def attempt():
            n = user_entry.get().strip()
            p = pass_entry.get().strip()
            if not n or not p:
                info.config(text="Completa ambos campos.")
                return
            if n in self.scores:
                if self.scores[n]["pass"] != sha(p):
                    info.config(text="Contraseña incorrecta.")
                    return
                result.append(n); root.destroy()
            else:
                if messagebox.askyesno("Registro", f"Crear usuario «{n}»?"):
                    self.scores[n] = {"pass": sha(p), "elo": 1200}
                    save_scores(self.scores)
                    result.append(n); root.destroy()

        tk.Button(root, text="Iniciar sesión", width=15, command=attempt).pack(pady=6)
        root.bind("<Return>", lambda *_: attempt())
        root.mainloop()

        return result[0] if result else None

    # ------------------------------------------------------------------
    # Menú de bienvenida
    # ------------------------------------------------------------------
    def _welcome_screen(self):
        w = tk.Tk(); w.title("MMC Chess – Bienvenido")
        tk.Label(w, text=f"¡Hola {self.player}! ¿Qué quieres hacer?", font=("Arial", 14)).pack(pady=10)
        tk.Button(w, text="Jugar contra la IA", width=20, command=lambda: self._start_game(w)).pack(pady=5)
        tk.Button(w, text="Ver IA vs IA",     width=20, command=lambda: self._start_selfplay(w)).pack(pady=5)
        w.protocol("WM_DELETE_WINDOW", lambda: (self.mmc.close(), w.destroy()))
        w.mainloop()

    # ------------------------------------------------------------------
    # Inicio de partida Humano vs IA
    # ------------------------------------------------------------------
    def _start_game(self, welcome):
        welcome.destroy()

        self.root = tk.Tk()
        self.root.title("MMC Chess")
        self.status = tk.StringVar(value="")
        top = tk.Frame(self.root); top.pack(fill="x")
        tk.Label(top, text="IA:").pack(side="left")
        tk.Label(top, textvariable=self.status).pack(side="left")

        self.board = chess.Board()
        self.sel = None
        self._load_imgs()
        self._build_menu()
        self._draw_board()

        self.root.after(60, self._poll_queue)
        threading.Thread(target=self._maint_loop, daemon=True).start()
        self.root.mainloop()

    # ------------------------------------------------------------------
    # IA vs IA en ventana aparte
    # ------------------------------------------------------------------
    def _start_selfplay(self, welcome):
        welcome.destroy()
        self.root = tk.Tk(); self.root.title("IA vs IA – MMC Chess")
        self.canvas = tk.Canvas(self.root, width=480, height=480); self.canvas.pack()
        self._load_imgs()
        threading.Thread(target=self._selfplay_loop, daemon=True).start()
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Recursos gráficos
    # ------------------------------------------------------------------
    def _load_imgs(self):
        self.imgs = {}
        sz = 60
        for c in "wb":
            for p in "prnbqk":
                fp = os.path.join(IMAGES_DIR, f"{c}{p}.png")
                if os.path.exists(fp):
                    self.imgs[c + p] = Image.open(fp).resize((sz, sz), Resampling.LANCZOS)

    def _build_menu(self):
        mb = tk.Menu(self.root)

        pm = tk.Menu(mb, tearoff=0)
        pm.add_command(label="Reiniciar partida", command=self._reset)
        pm.add_command(label="Ver IA vs IA…",    command=self._open_selfplay_window)
        pm.add_command(label="Reset memoria IA", command=self._reset_mmc)
        mb.add_cascade(label="Play", menu=pm)

        sm = tk.Menu(mb, tearoff=0)
        sm.add_command(label="Tabla de posiciones", command=self._show_scores)
        mb.add_cascade(label="Scores", menu=sm)

        am = tk.Menu(mb, tearoff=0)
        am.add_command(label="Clusters MMC", command=self._show_clusters)
        mb.add_cascade(label="Analyze", menu=am)

        self.root.config(menu=mb)
        self.root.protocol("WM_DELETE_WINDOW", lambda: (self.mmc.close(), self.root.destroy()))

    # ------------------------------------------------------------------
    # Dibujo del tablero
    # ------------------------------------------------------------------
    def _draw_board(self, canvas=None, board=None):
        if not hasattr(self, "canvas"):
            self.canvas = tk.Canvas(self.root, width=480, height=480)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self._click)
        if canvas is None:
            canvas, board = self.canvas, self.board

        img = Image.new("RGBA", (480, 480))
        draw = ImageDraw.Draw(img)
        clr = ("#F0D9B5", "#B58863"); sz = 60
        for r in range(8):
            for f in range(8):
                x, y = f * sz, (7 - r) * sz
                draw.rectangle([x, y, x + sz, y + sz], fill=clr[(f + r) & 1])
                pc = board.piece_at(chess.square(f, r))
                if pc:
                    k = ("w" if pc.color else "b") + pc.symbol().lower()
                    if k in self.imgs:
                        img.paste(self.imgs[k], (x, y), self.imgs[k])

        photo = ImageTk.PhotoImage(img); canvas.photo = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # ------------------------------------------------------------------
    # Click del usuario
    # ------------------------------------------------------------------
    def _click(self, e):
        f, r = e.x // 60, 7 - e.y // 60
        sq = chess.square(f, r)
        if self.sel is None:
            self.sel = sq; return
        mv = chess.Move(self.sel, sq)

        # promoción
        if self.board.piece_type_at(self.sel) == chess.PAWN and chess.square_rank(sq) in (0, 7):
            promo = simpledialog.askstring("Promoción", "q,r,b,n:", initialvalue="q")
            mp = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
            if promo in mp:
                mv = chess.Move(self.sel, sq, promotion=mp[promo])

        if mv in self.board.legal_moves:
            self.agent.record(self.board, mv)
            self.board.push(mv)
            self.agent.record(self.board)
            self._draw_board(); self.sel = None
            if self.board.is_game_over(claim_draw=True):
                self._end(); return
            threading.Thread(target=self._ia_move, daemon=True).start()
        else:
            messagebox.showwarning("Inválido", "Movimiento ilegal."); self.sel = None

    # ------------------------------------------------------------------
    # Movimiento de la IA en hilo aparte
    # ------------------------------------------------------------------
    def _ia_move(self):
        mv = self.agent.select(self.board)
        self.queue.put(mv)

    # ------------------------------------------------------------------
    # Sondeo de la cola IA -> GUI
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                mv = self.queue.get_nowait()
                self.agent.record(self.board, mv)
                self.board.push(mv)
                self.agent.record(self.board)
                self._draw_board()
                if self.board.is_game_over(claim_draw=True):
                    self._end()
        except queue.Empty:
            pass
        self.root.after(40, self._poll_queue)

    # ------------------------------------------------------------------
    # Mantenimiento (decay) cada minuto
    # ------------------------------------------------------------------
    def _maint_loop(self):
        while True:
            time.sleep(60)
            self.mmc.decay()

    # ------------------------------------------------------------------
    # Bucle de auto‑juego IA vs IA
    # ------------------------------------------------------------------
    def _selfplay_loop(self):
        a, b = Agent(self.mmc), Agent(self.mmc)
        board = chess.Board()
        opening_random_moves = 4
        while True:
            if board.is_game_over(claim_draw=True):
                res = board.result()
                if res in ("1-0", "0-1"):
                    winner = a if res == "1-0" else b
                    loser  = b if res == "1-0" else a
                    winner.finalize(1, 0); loser.finalize(0, 1)
                else:
                    a.finalize(0, 0); b.finalize(0, 0)
                board = chess.Board(); a.prev_fen = b.prev_fen = None
                opening_random_moves = 4

            ag = a if board.turn else b
            if opening_random_moves > 0:
                mv = random.choice(list(board.legal_moves))
                opening_random_moves -= 1
            else:
                mv = ag.select(board)

            ag.record(board, mv)
            board.push(mv)
            ag.record(board)

            self._draw_board(self.canvas, board)
            time.sleep(THINK_MS / 1000)

    # ------------------------------------------------------------------
    # Funciones varias de menú
    # ------------------------------------------------------------------
    def _reset(self):
        self.board.reset(); self.agent.prev_fen = None; self._draw_board()

    def _reset_mmc(self):
        self.mmc.close(); self.mmc = MMC(); self.agent.mmc = self.mmc

    def _show_scores(self):
        tbl = sorted(((d["elo"], n) for n, d in self.scores.items() if n != MACHINE_NAME), reverse=True)
        messagebox.showinfo(
            "Posiciones",
            "\n".join(f"{i+1}. {n} – {elo}" for i, (elo, n) in enumerate(tbl[:20])) or "Vacío",
        )

    def _show_clusters(self):
        G, c = self.mmc.snapshot()
        if not G.number_of_nodes():
            messagebox.showinfo("Clusters", "Sin datos suficientes aún."); return
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 6))
        for nodes in c.values():
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=15)
        plt.title("Clusters MMC"); plt.axis("off"); plt.show()

    def _open_selfplay_window(self):
        win = tk.Toplevel(self.root); win.title("IA vs IA")
        canvas = tk.Canvas(win, width=480, height=480); canvas.pack()
        w, b = Agent(self.mmc), Agent(self.mmc); board = chess.Board()

        def step():
            nonlocal board
            if board.is_game_over(claim_draw=True):
                board = chess.Board(); w.prev_fen = b.prev_fen = None
            ag = w if board.turn else b
            mv = ag.select(board); ag.record(board, mv); board.push(mv); ag.record(board)
            self._draw_board(canvas, board); win.after(THINK_MS, step)

        step()

    # ------------------------------------------------------------------
    # Fin de partida humano vs IA
    # ------------------------------------------------------------------
    def _end(self):
        res = self.board.result()
        sa = 1 if res == "1-0" else 0 if res == "0-1" else .5

        if sa == 1:
            self.agent.finalize(1, 0)
        elif sa == 0:
            self.agent.finalize(0, 1)

        h, m = self.scores[self.player]["elo"], self.scores[MACHINE_NAME]["elo"]
        nh, nm = elo_update(h, m, sa)
        self.scores[self.player]["elo"], self.scores[MACHINE_NAME]["elo"] = round(nh), round(nm)
        save_scores(self.scores)

        game = chess.pgn.Game.from_board(self.board)
        self.mmc.store_game(str(game), res, self.player)

        msg = "Ganaste" if sa == 1 else "Empate" if sa == 0.5 else "Perdiste"
        if messagebox.askyesno("Fin", f"{msg}\n¿Revancha?"):
            self._reset()
        else:
            self.root.destroy()


# ---------------------------------------------------------------------------
# --- HOTFIX PATCH (24‑may‑2025) -----------------------------
# Corrige indentaciones faltantes en v47. Se enlazan los métodos a
# sus clases respectivas para evitar AttributeError y para que MMC
# tenga 'choose', 'decay', 'snapshot', 'store_game', 'import_pgn'.
# Además se conecta Agent.finalize / select / _mcts / evaluate.

import random, math, time, collections, os
import networkx
import chess, mysql.connector

def _mmc_choose(self, fen: str, creative: bool = False):
    rows = self.q(
        "SELECT dst, weight, capture, `escape`, win, loss FROM edges WHERE src=%s",
        (fen,),
    ).fetchall()
    if not rows:
        return None
    scores = []
    for dst, w, cap, esc, win, loss in rows:
        wl_ratio = (win + 1) / (win + loss + 2)     # suavizado
        score = (w + 1e-3) * (1 + 0.75*cap - 0.40*esc) * (0.5 + wl_ratio)
        scores.append(score)
    best_idx = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_idx]
    avg = sum(scores) / len(scores)
    if not creative and best_score > 1.2*avg and random.random() > self.exploration:
        return rows[best_idx][0]
    return random.choices([r[0] for r in rows], weights=scores, k=1)[0]

def _mmc_decay(self):
    f = math.exp(-self.decay_rate * 60)
    self.q("UPDATE edges SET weight = weight * %s WHERE win = 0", (f,))
    self.q(
        "DELETE FROM edges WHERE weight < %s AND win = 0 AND loss > 5",
        (self.base_prune,),
    )

def _mmc_snapshot(self):
    G = networkx.Graph()
    G.add_edges_from(self.q("SELECT src,dst FROM edges LIMIT 3000").fetchall())
    comms = (
        networkx.algorithms.community.louvain_communities(G, seed=42)
        if G.number_of_nodes()
        else []
    )
    return G, {i: list(c) for i, c in enumerate(comms)}

def _mmc_store_game(self, pgn: str, res: str, player: str):
    self.enqueue(
        "INSERT INTO games(pgn,result,player) VALUES(%s,%s,%s)",
        (pgn, res, player),
    )

def _mmc_import_pgn(self, path: str, player: str = "import") -> int:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    n_games = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            self.store_game(str(game), result, player)
            board = game.board()
            for mv in game.mainline_moves():
                from_fen = canon(board)
                capture = 1 if board.is_capture(mv) else 0
                escape_before = board.is_check()
                board.push(mv)
                to_fen = canon(board)
                winner_is_white = result == "1-0"
                color_to_move = not board.turn
                win = int((color_to_move and winner_is_white) or (not color_to_move and not winner_is_white))
                loss = int((color_to_move and not winner_is_white) or (not color_to_move and winner_is_white))
                self.update(
                    from_fen,
                    to_fen,
                    0.05,
                    capture=capture,
                    escape_=int(escape_before),
                    win=win,
                    loss=loss,
                )
            n_games += 1
    return n_games

# Realizar binding de los métodos a la clase MMC
MMC.choose = _mmc_choose
MMC.decay = _mmc_decay
MMC.snapshot = _mmc_snapshot
MMC.store_game = _mmc_store_game
MMC.import_pgn = _mmc_import_pgn

# Conectar métodos que quedaron fuera de Agent
Agent.finalize = finalize
Agent.select   = select
Agent._mcts    = _mcts
Agent.evaluate = evaluate
# ------------------------------------------------------------

# --- EVALUATE FIX (24‑may‑2025) -----------------------------
DRAW_PENALTY = 0.0

def evaluate(self, board: chess.Board) -> float:
    """Evaluación estática rápida a partir de la tabla *features*.
    Devuelve una puntuación desde la perspectiva del jugador que está por mover
    (positiva = bueno para el bando al turno, negativa = malo)."""
    # Finales inmediatos
    if board.is_checkmate():
        return -9999.0  # perdió el que mueve
    if board.is_stalemate():
        return DRAW_PENALTY
    if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return DRAW_PENALTY

    row = self.mmc.get_features(canon(board))
    if not row:
        # Sin datos: usar heurística material mínima
        val = sum(map_piece(piece) for piece in board.piece_map().values())
        return val if board.turn else -val

    mat_diff, at_w, at_b, de_w, de_b, ks_w, ks_b, mob_w, mob_b = row
    sign = 1 if board.turn else -1

    mat  = sign * mat_diff
    atk  = (at_w - at_b) if board.turn else (at_b - at_w)
    dfn  = (de_w - de_b) if board.turn else (de_b - de_w)
    ks   = (ks_b - ks_w) if board.turn else (ks_w - ks_b)
    mob  = (mob_w - mob_b) if board.turn else (mob_b - mob_w)

    val = 10*mat + 0.5*atk - 0.3*dfn + 0.8*ks + 0.1*mob
    return val
# Vincular la versión correcta
Agent.evaluate = evaluate
# ------------------------------------------------------------


# --- HOTFIX PATCH 2 (24‑may‑2025) -----------------------------
# Soluciona NameError 'map_piece', arregla sangría y completa lógica
# de evaluate con función material fallback.

def _piece_value(piece: chess.Piece) -> float:
    table = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
             chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
    return table.get(piece.piece_type, 0.0)

def evaluate(self, board: chess.Board) -> float:
    # Fin de partida
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate() or board.can_claim_threefold_repetition()        or board.can_claim_fifty_moves():
        return DRAW_PENALTY

    # Buscar features almacenadas
    row = self.mmc.get_features(canon(board))
    if row:
        (mat_diff, at_w, at_b, de_w, de_b, ks_w, ks_b, mob_w, mob_b) = row
        sign = 1 if board.turn else -1
        mat = sign * mat_diff
        atk = (at_w - at_b)
        dfn = (de_w - de_b)
        ks  = sign * (ks_w - ks_b)
        mob = sign * (mob_w - mob_b)
        return 10*mat + 0.5*atk - 0.3*dfn + 0.8*ks + 0.1*mob

    # Fallback: material simple
    val = sum(_piece_value(p) for p in board.piece_map().values())
    return val if board.turn else -val

# Re‑vincular
Agent.evaluate = evaluate
# ------------------------------------------------------------

# ---------- FAST‑PLAY PATCH (25‑may‑2025) -------------------
# Objetivo: acelerar la toma de decisiones sin sacrificar calidad.
# Estrategia:
#   1. Límite de tiempo por jugada (TIME_BUDGET).
#   2. Simulaciones MCTS adaptativas [MIN_SIMS .. MAX_SIMS].
#   3. Reutilización de árbol entre turnos (self._root_dict).
#   4. Abort early si memoria indica una jugada con >70 % del score.

TIME_BUDGET = 0.25      # seg por jugada (ajustable)
MIN_SIMS    = 40
MAX_SIMS    = 300
FORCE_MOVE_THRESHOLD = 0.70  # confianza memoria

# Caché de árboles por posición FEN (solo en RAM)
Agent._root_dict = {}

def select(self, board: chess.Board):
    fen = canon(board)
    # 1) Intenta usar memoria (edges) con alta confianza
    choice = self.mmc.choose(fen, creative=self.creative)
    if choice:
        dst_fen, conf = choice
        if conf >= FORCE_MOVE_THRESHOLD:
            for mv in board.legal_moves:
                board.push(mv)
                if canon(board) == dst_fen:
                    board.pop()
                    return mv
                board.pop()

    # 2) Recuperar root si existe
    root = self._root_dict.get(fen)
    if root is None:
        root = Node(fen)  # Node definido en tu _mcts
        self._root_dict[fen] = root

    # 3) MCTS dentro del presupuesto de tiempo/simulaciones
    start = time.time()
    sims  = 0
    while (time.time() - start) < TIME_BUDGET and sims < MAX_SIMS:
        self._mcts(board, root)
        sims += 1
        if sims >= MIN_SIMS and root.best_ucb() > 1000:  # ventaja decisiva
            break

    # 4) Elegir mejor hijo
    best_move = root.best_move()
    return best_move

# Enlazar nueva versión
Agent.select = select
# ------------------------------------------------------------

# ---------- FAST‑PLAY PATCH 2 (25‑may‑2025) -------------
# 1) Corrige AttributeError 'creative'
# 2) Hace que MMC.choose devuelva (dst_fen, confidence)

# Garantizar atributo 'creative' en Agent
if not hasattr(Agent, "creative"):
    Agent.creative = False  # valor por defecto

# Nueva versión de choose con confianza
def _mmc_choose_conf(self, fen: str, creative: bool = False):
    rows = self.q(
        "SELECT dst, weight, capture, `escape`, win, loss FROM edges WHERE src=%s",
        (fen,),
    ).fetchall()
    if not rows:
        return None
    scores = []
    for dst, w, cap, esc, win, loss in rows:
        wl_ratio = (win + 1) / (win + loss + 2)
        score = (w + 1e-3) * (1 + 0.75*cap - 0.40*esc) * (0.5 + wl_ratio)
        scores.append(score)
    total = sum(scores)
    best_idx = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_idx]
    conf = best_score / total if total else 0.0
    if not creative and conf >= 0.60:
        return rows[best_idx][0], conf
    # sample
    dst = random.choices([r[0] for r in rows], weights=scores, k=1)[0]
    sel_idx = [r[0] for r in rows].index(dst)
    return dst, scores[sel_idx]/total if total else 0.0

# Re‑binding
MMC.choose = _mmc_choose_conf

# Actualizar select para usar getattr
def select(self, board: chess.Board):
    fen = canon(board)
    choice = self.mmc.choose(fen, creative=getattr(self,"creative",False))
    if choice:
        dst_fen, conf = choice
        if conf >= FORCE_MOVE_THRESHOLD:
            for mv in board.legal_moves:
                board.push(mv)
                if canon(board) == dst_fen:
                    board.pop()
                    return mv
                board.pop()

    # resto igual que antes
    root = self._root_dict.get(fen)
    if root is None:
        root = Node(fen)
        self._root_dict[fen] = root
    start = time.time()
    sims  = 0
    while (time.time() - start) < TIME_BUDGET and sims < MAX_SIMS:
        self._mcts(board, root)
        sims += 1
        if sims >= MIN_SIMS and root.best_ucb() > 1000:
            break
    return root.best_move()

Agent.select = select
# ---------------------------------------------------------

# ---------- FAST‑PLAY PATCH 3 (25‑may‑2025) -------------
# Reescribe Agent.select para no depender de Node.

SIMS_BATCH = 60  # playouts por llamada a _mcts

def select(self, board: chess.Board):
    fen = canon(board)
    choice = self.mmc.choose(fen, creative=getattr(self, "creative", False))
    if choice:
        dst_fen, conf = choice
        if conf >= FORCE_MOVE_THRESHOLD:
            for mv in board.legal_moves:
                board.push(mv)
                if canon(board) == dst_fen:
                    board.pop()
                    return mv
                board.pop()

    # Presupuesto de tiempo / simulaciones con _mcts existente
    start = time.time()
    sims_done = 0
    last_move = None
    while sims_done < MAX_SIMS and (time.time() - start) < TIME_BUDGET:
        last_move = self._mcts(board, sims=SIMS_BATCH)
        sims_done += SIMS_BATCH
        if sims_done >= MIN_SIMS:
            break
    return last_move

Agent.select = select
# --------------------------------------------------------

# -------------- AUTO‑OPTIMIZATION PATCH (25‑may‑2025) ----------------
# Añade poda inteligente, hibernación reactiva y auto‑optimizador.

import threading, time

# ---------- 1. Extender MMC.__init__ para preparar tablas ------------
_old_init = MMC.__init__
def _init_with_cleanup(self, *args, **kwargs):
    _old_init(self, *args, **kwargs)
    # Crear tabla de archivo si no existe
    self.q("CREATE TABLE IF NOT EXISTS edges_archive LIKE edges")
    # Añadir columna stale si falta
    cols = {r[0] for r in self.q("SHOW COLUMNS FROM edges").fetchall()}
    if 'stale' not in cols:
        self.q("ALTER TABLE edges ADD COLUMN stale TINYINT DEFAULT 0")
    self.played = 0
    self._last_cleanup = time.time()
MMC.__init__ = _init_with_cleanup

# ---------- 2. Poda y compactación -----------------------------------
def compact_db(self):
    # Marcar aristas candidatas
    self.q("""UPDATE edges
              SET stale = 1
              WHERE weight < 0.02
                AND win + loss = 0
                AND last_update < NOW() - INTERVAL 30 DAY""")
    # Mover a archivo
    self.q("""INSERT INTO edges_archive
              SELECT * FROM edges WHERE stale = 1""")
    self.q("DELETE FROM edges WHERE stale = 1")
    # Eliminar nodos huérfanos
    self.q("""DELETE n FROM nodes n
              LEFT JOIN edges e ON e.src = n.fen OR e.dst = n.fen
              WHERE e.src IS NULL AND e.dst IS NULL""")
    self.q("ANALYZE TABLE edges, nodes")
MMC.compact_db = compact_db

# ---------- 3. Reactivar desde archivo -------------------------------
def revive_if_needed(self, src_fen, dst_fen):
    row = self.q("""SELECT src,dst,weight,capture,escape,win,loss
                    FROM edges_archive
                    WHERE src=%s AND dst=%s""",
                 (src_fen, dst_fen)).fetchone()
    if row:
        src, dst, w, cap, esc, win, loss = row
        self.q("""INSERT INTO edges(src,dst,weight,capture,escape,win,loss)
                  VALUES(%s,%s,%s,%s,%s,%s,%s)
                  ON DUPLICATE KEY UPDATE weight=weight+VALUES(weight)""",
               (src, dst, w*0.5, cap, esc, win, loss))
        self.q("DELETE FROM edges_archive WHERE src=%s AND dst=%s",
               (src_fen, dst_fen))
        return True
    return False
MMC.revive_if_needed = revive_if_needed

# ---------- 4. Auto‑optimizador online -------------------------------
def auto_optimize(self):
    if self.played % 10000 == 0:
        self.q("""UPDATE edges
                  SET weight = weight * 0.9
                  WHERE last_update < NOW() - INTERVAL 7 DAY""")
    if time.time() - self._last_cleanup > 86400:
        threading.Thread(target=self.compact_db, daemon=True).start()
        self._last_cleanup = time.time()
MMC.auto_optimize = auto_optimize

# ---------- 5. Hookear en choose y finalize --------------------------
_prev_choose = MMC.choose
def _choose_with_revival(self, fen, creative=False):
    result = _prev_choose(self, fen, creative)
    if result is None:
        rows = self.q("""SELECT dst FROM edges_archive WHERE src=%s LIMIT 1""", (fen,)).fetchall()
        if rows:
            dst = rows[0][0]
            self.revive_if_needed(fen, dst)
            result = _prev_choose(self, fen, creative)
    return result
MMC.choose = _choose_with_revival

if hasattr(Agent, "finalize"):
    old_finalize = Agent.finalize
    def _fin_and_track(self, *a, **kw):
        old_finalize(self, *a, **kw)
        self.mmc.played += 1
        self.mmc.auto_optimize()
    Agent.finalize = _fin_and_track
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ChessGUI()