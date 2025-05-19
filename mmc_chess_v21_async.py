
# -*- coding: utf-8 -*-
"""
MMC Chess Learner – v10.1  (10‑may‑2025)
-----------------------------------------------------------
* Soluciona error 1054 «Unknown column 'win' in 'field list'».
  – Si la tabla edges proviene de versiones antiguas, ahora se
    agregan las columnas faltantes (win, loss) en _schema().
* Añade verificación dinámica de columnas y ALTER TABLE en caliente.
* No cambia la lógica anterior.
"""

from __future__ import annotations
import tkinter as tk, threading, time, random, json, os, math, hashlib, queue, collections
from tkinter import filedialog, messagebox, simpledialog
import mysql.connector, chess, chess.pgn
from PIL import Image, ImageTk, ImageDraw
from PIL.Image import Resampling
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities

IMAGES_DIR, SCORE_FILE = "images", "scoreboard.json"
MACHINE_NAME, THINK_MS = "_machine_", 40
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()

def elo_exp(ra, rb): return 1/(1+10**((rb-ra)/400))
def elo_update(ra, rb, sa, k=32): return ra+k*(sa-elo_exp(ra,rb)), rb+k*((1-sa)-elo_exp(rb,ra))

def reward(prev: chess.Board, new: chess.Board, mv: chess.Move):
    r=.10
    if prev.is_capture(mv): r+=.5
    if prev.is_check() and not new.is_check(): r+=.4
    if new.is_check(): r+=.2
    if new.is_checkmate(): r+=1
    return r

def load_scores():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE,'r',encoding='utf-8') as f:
            return json.load(f)
    return {}
def save_scores(s): open(SCORE_FILE,'w',encoding='utf-8').write(json.dumps(s,indent=2))

# ---------------- MMC -----------------
class MMC:
    def __init__(self, cfg="db_config.json"):
        self.cfg = json.load(open(cfg, "r", encoding="utf-8"))

        db_name = self.cfg.get("database", "mmc_chess")  # nombre deseado

        cfg_nodb = {k: v for k, v in self.cfg.items() if k != "database"}
        tmp = mysql.connector.connect(**cfg_nodb, autocommit=True)
        cur = tmp.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cur.close()
        tmp.close()
                # 2) vuelve a conectar ya dentro de la nueva BD
        self.conn = mysql.connector.connect(**self.cfg, autocommit=True)
        self.write_q = queue.Queue()
        threading.Thread(target=self._flush_worker, daemon=True).start()
        self.exploration, self.lateral_eps=.25,.20
        self.decay_rate, self.base_prune=.001,.50
        self._schema()
        self.last_fens=collections.deque(maxlen=32)
        cur0=self.conn.cursor()
        cur0.execute("""CREATE DATABASE IF NOT EXISTS mmc_chess""")
        # -------------------- MySQL helpers ------------------------
       
        #  --- PEGAR AQUÍ ---
    def update(self, src_fen, dst_fen, delta,
               capture=0, escape=0, win=0, loss=0):
        """
        Refuerza la arista (src_fen → dst_fen) en la memoria MMC.

        Parameters
        ----------
        src_fen : str   Posición origen en FEN
        dst_fen : str   Posición destino en FEN
        delta   : float Recompensa (+) o castigo (–)
        capture : int   1 si fue captura, 0 si no
        escape  : int   1 si se salió de jaque, 0 si no
        win     : int   1 si la partida terminó ganada, 0 en otro caso
        loss    : int   1 si la partida terminó perdida, 0 en otro caso
        """
        now = time.time()

        # actualiza / inserta nodos
        self.q("""INSERT INTO nodes VALUES(%s,%s)
                  ON DUPLICATE KEY UPDATE last_update=%s""",
               (src_fen, now, now))
        self.q("""INSERT INTO nodes VALUES(%s,%s)
                  ON DUPLICATE KEY UPDATE last_update=%s""",
               (dst_fen, now, now))

        # refuerza arista
        self.q("""INSERT INTO edges(src,dst,weight,win,loss,capture,escape)
                    VALUES(%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      weight  = weight  + %s,
                      win     = win     + %s,
                      loss    = loss    + %s,
                      capture = capture | %s,
                      escape  = escape  | %s""",
               (src_fen, dst_fen, delta, win, loss, capture, escape,
                delta, win, loss, capture, escape))
    #  --- FIN update() ---
    def _schema(self):
        cur=self.conn.cursor()
        cur.execute("""CREATE DATABASE IF NOT EXISTS mmc_chess""")
        cur.execute("""CREATE TABLE IF NOT EXISTS nodes(
        fen VARCHAR(120) PRIMARY KEY,
        last_update DOUBLE)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS edges(
        src VARCHAR(120), dst VARCHAR(120),
        weight DOUBLE DEFAULT 0,
        capture TINYINT DEFAULT 0,
        escape  TINYINT DEFAULT 0,
        PRIMARY KEY(src,dst),
        INDEX(src), INDEX(dst))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS games(
        id INT AUTO_INCREMENT PRIMARY KEY,
        pgn MEDIUMTEXT,
        result VARCHAR(7),
        player VARCHAR(64),
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        # --- migración a v10 ---
        # ver columnas actuales
        cur.execute("SHOW COLUMNS FROM edges")
        cols={r[0] for r in cur.fetchall()}
        migrate=[]
        if 'win' not in cols:
            migrate.append("ADD COLUMN win INT DEFAULT 0")
        if 'loss' not in cols:
            migrate.append("ADD COLUMN loss INT DEFAULT 0")
        # si faltan, alter table
        if migrate:
            cur.execute("ALTER TABLE edges " + ", ".join(migrate))
        

    def _get_conn(self):
        """Devuelve (y cachea en thread‑local) una conexión viva y exclusiva
        para el hilo actual. Evita usar el mismo objeto conn desde varios hilos,
        requisito de mysql‑connector y evita los ping fallidos."""
        if not hasattr(self,'_local'):
            self._local = threading.local()
        cn = getattr(self._local,'conn',None)
        if cn is None or not cn.is_connected():
            cn = mysql.connector.connect(**self.cfg,autocommit=True)
            self._local.conn = cn
        return cn

    def q(self, sql, params=()):
        """Obtiene un cursor sobre una conexión *por hilo*.
        El llamador debe cerrar el cursor tras fetch.*(), manteniendo la interfaz
        original y evitando conflictos de threads con MySQL."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute(sql, params)
        return cur

        self.enqueue(sql_nodes, (a, now, now))
        self.enqueue(sql_nodes, (b, now, now))
        # aristas
        sql_edges = """INSERT INTO edges(src,dst,weight,win,loss,capture,escape)
                         VALUES(%s,%s,%s,%s,%s,%s,%s)
                         ON DUPLICATE KEY UPDATE
                           weight = weight + %s,
                           win    = win + %s,
                           loss   = loss + %s,
                           capture = capture | %s,
                           escape  = escape  | %s"""
        self.enqueue(sql_edges, (a, b, delta, win, loss, capture, escape,
                                 delta, win, loss, capture, escape))

    # --- Async write machinery ---
    def enqueue(self, sql, params):
        """Put an INSERT/UPDATE in the write queue."""
        self.write_q.put((sql, params))

    def _flush_worker(self):
        cur = self.conn.cursor()
        pending = []
        last_commit = time.time()
        while True:
            try:
                sql, params = self.write_q.get(timeout=0.5)
                cur.execute(sql, params)
                pending.append(1)
                self.write_q.task_done()
            except queue.Empty:
                pass
            # commit batch every 0.5 s or 100 stmts
            if pending and (len(pending) >= 100 or time.time() - last_commit > 0.5):
                self.conn.commit()
                pending.clear()
                last_commit = time.time()
    # selección de destino

    def get_edge(self, src, dst):
        """Devuelve la fila de edges (o None)."""
        return self.q("SELECT src,dst,weight,win,loss,capture,escape FROM edges WHERE src=%s AND dst=%s", (src,dst)).fetchone()
    def choose(self, fen: str, creative=False):
        rows = self.q("SELECT dst, weight, capture, escape, loss FROM edges WHERE src=%s",
                      (fen,)).fetchall()
        if not rows: return None

        moves, base_w, caps, escs, losses = zip(*rows)
        # rebaja por derrotas
        weights = [ max(0.001,
                        w * (1 + 0.15*c + 0.10*e) * (0.5**min(l,3)) )
                    for w,c,e,l in zip(base_w,caps,escs,losses) ]

        # + novedad contextual
        if fen in self.last_fens:
            self.lateral_eps = min(.6, self.lateral_eps + .05)
        else:
            self.lateral_eps = max(.10, self.lateral_eps - .02)

        # creatividad apertura
        if creative or random.random() < self.exploration:
            return random.choices(moves, k=1)[0]

        # pensamiento lateral
        if random.random() < self.lateral_eps:
            return moves[weights.index(min(weights))]

        # ponderado
        total = sum(weights)
        pick = random.random() * total
        acc = 0
        for m,w in zip(moves,weights):
            acc += w
            if pick <= acc:
                return m
        return moves[-1]

    def decay(self):
        f = math.exp(-self.decay_rate*60)
        self.q("UPDATE edges SET weight = weight * %s WHERE win = 0", (f,))
        self.q("DELETE FROM edges WHERE weight < %s AND win=0 AND loss>5", (self.base_prune,))

    def snapshot(self):
        G = nx.Graph()
        G.add_edges_from(self.q("SELECT src,dst FROM edges LIMIT 3000").fetchall())
        comms = louvain_communities(G, seed=42) if G.number_of_nodes() else []
        return G, {i:list(c) for i,c in enumerate(comms)}

    def store_game(self, pgn, res, player): 
        self.q("INSERT INTO games(pgn,result,player) VALUES(%s,%s,%s)",
               (pgn,res,player))


    def import_pgn(self, path, player="import"):
        """Importa un archivo PGN y alimenta la MMC con cada partida.

        Parameters
        ----------
        path : str
            Ruta al archivo .pgn
        player : str
            Nombre de jugador que se asociará a las partidas importadas.
        Returns
        -------
        int
            Cantidad de partidas procesadas.
        """
        import chess.pgn, os, datetime
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        n_games = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                # Guardar pgn completo
                self.store_game(str(game), result, player)
                # Recorrer jugadas para reforzar MMC
                board = game.board()
                for mv in game.mainline_moves():
                    from_fen = board.fen()
                    capture = 1 if board.is_capture(mv) else 0
                    escape_before = board.is_check()
                    board.push(mv)
                    to_fen = board.fen()
                    # Determinar win/loss desde la perspectiva de 'player'
                    win = 1 if result == "1-0" else 0
                    loss = 1 if result == "0-1" else 0
                    # Pequeña recompensa base
                    self.update(from_fen, to_fen, 0.05, capture=capture, escape=1 if escape_before else 0,
                                win=win, loss=loss)
                n_games += 1
        return n_games

    def close(self): self.conn.close()

# ---------- Agente ----------
class Agent:
    def __init__(self, mmc):
        self.mmc = mmc

        # Piece-Square Tables (simple 64‑element arrays for quick heuristic)
        # Order: white POV indices 0..63 (a1=0, h8=63). Values in centipawns.
        self.PST = {
            'P': [0,  5,  5, -5, -5,  5, 10,  0]*8,
            'N': [-50, -40, -30, -30, -30, -30, -40, -50]*8,
            'B': [-20, -10, -10, -10, -10, -10, -10, -20]*8,
            'R': [0, 0, 0, 5, 5, 0, 0, 0]*8,
            'Q': [-10, -5, -5, -5, -5, -5, -5, -10]*8,
            'K': [-30, -40, -40, -50, -50, -40, -40, -30]*8
        }
        # material values for quick evaluation
        self.MAT = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
        self.prev_fen = None
        self.prev_board = None
        self.path: list[tuple[str,str,int,int]] = []   # (a,b,capture,escape)

    def record(self, b: chess.Board, mv: chess.Move|None=None):
        fen = b.fen()
        if self.prev_fen:
            cap = int(self.prev_board.is_capture(mv)) if mv else 0
            esc = int(self.prev_board.is_check() and not b.is_check()) if mv else 0
            self.mmc.update(self.prev_fen, fen,
                            reward(self.prev_board,b,mv) if mv else .05,
                            cap, esc)
            self.path.append((self.prev_fen, fen, cap, esc))
        self.prev_fen, self.prev_board = fen, b.copy(stack=False)
        self.mmc.last_fens.append(fen)

    def select(self, b: chess.Board):
        candidate_moves=[]
        tgt=self.mmc.choose(b.fen(), b.fullmove_number<=4)
        if tgt:
            for mv in b.legal_moves:
                tmp=b.copy()
                tmp.push(mv)
                if tmp.fen()==tgt:
                    candidate_moves.append(mv)
        if not candidate_moves:
            candidate_moves=list(b.legal_moves)
        # avoid moves leading to recent positions
        recent=set(self.mmc.last_fens[-12:])
        filtered=[]
        for mv in candidate_moves:
            tmp=b.copy()
            tmp.push(mv)
            if tmp.fen() not in recent:
                filtered.append(mv)
        if filtered:
            candidate_moves=filtered
        # score moves by material gain to increase aggressiveness
        best_mv=random.choice(candidate_moves)
        best_score=-1e9
        for mv in candidate_moves:
            tmp=b.copy(); tmp.push(mv)
            score=self.evaluate(tmp)
            if b.is_capture(mv):
                cap_piece=b.piece_at(mv.to_square)
                own_piece=b.piece_at(mv.from_square)
                if cap_piece:
                    score+= self.MAT[cap_piece.piece_type]*0.8
                if own_piece:
                    score-= self.MAT[own_piece.piece_type]*0.2
            if score>best_score:
                best_score=score; best_mv=mv
        return best_mv
    # refuerzo por resultado
    def finalize(self, win_flag: int, loss_flag:int):
        for a,b,cap,esc in self.path:
            self.mmc.update(a,b, 0.4 if win_flag else -0.4 if loss_flag else .0,
                            cap, esc, win_flag, loss_flag)
        self.path.clear()

# ---------- GUI ----------

    def select(self, board: chess.Board):
        """Elige el mejor movimiento usando un MCTS ligero con priors de la MMC."""
        # Si estamos en la fase de apertura (≤4), usa directamente la MMC
        if board.fullmove_number <= 4:
            tgt = self.mmc.choose(board.fen(), creative=False)
            if tgt:
                for mv in board.legal_moves:
                    b2 = board.copy(); b2.push(mv)
                    if b2.fen() == tgt:
                        return mv
        # De lo contrario ejecuta un MCTS limitado
        return self._mcts(board, sims=min(80, 20 + len(list(board.legal_moves))*2))

    def _mcts(self, board: chess.Board, sims:int=120):
        import math, random
        root_fen = board.fen()
        children = {root_fen: []}
        N = collections.Counter()   # visitas por estado
        W = collections.Counter()   # suma de valores
        P = {}                      # prior

        def expand(fen):
            if fen in children and children[fen]: return
            moves = list(chess.Board(fen).legal_moves)
            priors = []
            for mv in moves:
                b2 = chess.Board(fen); b2.push(mv)
                tgt = b2.fen()
                p = 1.0
                # prior from DB
                row = self.mmc.get_edge(fen, tgt)
                if row:
                    _,_,w,win,loss,cap,esc = row
                    p = w + 0.5*win - 0.3*loss + 0.1*cap - 0.1*esc + 1e-3
                priors.append((mv, tgt, p))
            s = sum(p for _,_,p in priors)
            priors = [(mv, tgt, p/s) for mv,tgt,p in priors]
            children[fen] = priors
            for _, tgt, p in priors:
                P[(fen, tgt)] = p

        def simulate():
            path = []
            fen = root_fen
            expand(fen)
            while True:
                # selección
                best, best_tgt, best_u = None, None, -1
                for mv, tgt, _ in children[fen]:
                    u = W[(fen,tgt)]/ (1+N[(fen,tgt)]) + 1.4 * P[(fen,tgt)] * math.sqrt(N[fen]+1)/(1+N[(fen,tgt)])
                    if u > best_u:
                        best_u, best, best_tgt = u, mv, tgt
                path.append((fen, best_tgt))
                fen = best_tgt
                if fen not in children:
                    expand(fen)
                if chess.Board(fen).is_game_over() or len(path)>4:
                    break
            # value: simple material eval
            val = self.evaluate(chess.Board(fen))
            # backprop
            for s,a in path:
                N[(s,a)] += 1
                W[(s,a)] += val

        for _ in range(sims):
            simulate()

        # escoger mejor movimiento
        best_mv, best_visits = None, -1
        for mv, tgt, _ in children[root_fen]:
            v = N[(root_fen, tgt)]
            if v > best_visits:
                best_visits = v
                best_mv = mv
        

        # devolver mejor movimiento (fallback aleatorio si no se simuló nada)
        return best_mv if best_mv is not None else random.choice(list(board.legal_moves))

    def evaluate(self, board: chess.Board):
        """Material evaluation with simple check bonus."""
        val = 0
        for piece in board.piece_map().values():
            val_piece = self.MAT[piece.piece_type]
            val += val_piece if piece.color==board.turn else -val_piece
        if board.is_check():
            val += 0.25
        return val
    def update(self, src_fen: str, dst_fen: str, w: float, cap: int = 0, esc: int = 0, win: int = 0, loss: int = 0):
        """Actualiza el grafo MMC con la transición src→dst."""
        now = time.time()
        sql_node = (
            "INSERT INTO nodes(fen,last_update) VALUES(%s,%s) "
            "ON DUPLICATE KEY UPDATE last_update=%s"
        )
        self.enqueue(sql_node, (src_fen, now, now))
        self.enqueue(sql_node, (dst_fen, now, now))

        sql_edge = (
            "INSERT INTO edges(src,dst,weight,win,loss,capture,escape) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE "
            "weight = weight + %s, win = win + %s, loss = loss + %s, "
            "capture = capture + %s, escape = escape + %s"
        )
        self.enqueue(sql_edge, (src_fen, dst_fen, w, win, loss, cap, esc,
                                w, win, loss, cap, esc))

class ChessGUI:
    def __init__(self, cfg="db_config.json"):
        self.scores=load_scores()
        self.scores.setdefault(MACHINE_NAME, {"pass":"", "elo":1500})
        self.player=self._login()
        if not self.player: return
        self.mmc=MMC()
        self.agent=Agent(self.mmc)
        self.queue=queue.Queue()
        self._welcome_screen()
   # ----- login -----
    def _login(self):
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
        result = []

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

    # ----- menú principal -----
    def _welcome_screen(self):
        w = tk.Tk(); w.title("MMC Chess – Bienvenido")
        tk.Label(w, text=f"¡Hola {self.player}! ¿Qué quieres hacer?", font=("Arial", 14)).pack(pady=10)
        tk.Button(w, text="Jugar contra la IA", width=20,
                  command=lambda: self._start_game(w)).pack(pady=5)
        tk.Button(w, text="Ver IA vs IA", width=20,
                  command=lambda: self._start_selfplay(w)).pack(pady=5)
        w.protocol("WM_DELETE_WINDOW", lambda: (self.mmc.close(), w.destroy()))
        w.mainloop()

    # ----- juego humano vs IA -----
    def _start_game(self, welcome):
        welcome.destroy()
        self.root = tk.Tk(); self.root.title("MMC Chess")
        self.status = tk.StringVar(value="")
        top = tk.Frame(self.root); top.pack(fill="x")
        tk.Label(top, text="IA:").pack(side="left")
        tk.Label(top, textvariable=self.status).pack(side="left")
        self.board = chess.Board(); self.sel = None
        self._load_imgs(); self._build_menu(); self._draw_board()
        self.root.after(60, self._poll_queue)
        threading.Thread(target=self._maint_loop, daemon=True).start()
        self.root.mainloop()

    # ----- self‑play -----
    def _start_selfplay(self, welcome):
        welcome.destroy()
        self.root = tk.Tk(); self.root.title("IA vs IA – MMC Chess")
        self.canvas = tk.Canvas(self.root, width=480, height=480); self.canvas.pack()
        self._load_imgs()
        threading.Thread(target=self._selfplay_loop, daemon=True).start()
        self.root.mainloop()

    # ----- recursos -----
    def _load_imgs(self):
        self.imgs = {}; sz = 60
        for c in "wb":
            for p in "prnbqk":
                fp = os.path.join(IMAGES_DIR, f"{c}{p}.png")
                if os.path.exists(fp):
                    self.imgs[c+p] = Image.open(fp).resize((sz, sz), Resampling.LANCZOS)

    def _build_menu(self):
        mb = tk.Menu(self.root)
        pm = tk.Menu(mb, tearoff=0)
        pm.add_command(label="Reiniciar partida", command=self._reset)
        pm.add_command(label="Ver IA vs IA…", command=self._open_selfplay_window)
        pm.add_command(label="Reset memoria IA", command=self._reset_mmc)
        mb.add_cascade(label="Play", menu=pm)
        sm = tk.Menu(mb, tearoff=0); sm.add_command(label="Tabla de posiciones", command=self._show_scores)
        mb.add_cascade(label="Scores", menu=sm)
        am = tk.Menu(mb, tearoff=0); am.add_command(label="Clusters MMC", command=self._show_clusters)
        mb.add_cascade(label="Analyze", menu=am)
        self.root.config(menu=mb)
        self.root.protocol("WM_DELETE_WINDOW", lambda: (self.mmc.close(), self.root.destroy()))

    # ----- dibujo tablero -----
    def _draw_board(self, canvas=None, board=None):
        if not hasattr(self, "canvas"):
            self.canvas = tk.Canvas(self.root, width=480, height=480)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self._click)
        if canvas is None:
            canvas, board = self.canvas, self.board
        img = Image.new("RGBA", (480, 480))
        draw = ImageDraw.Draw(img); clr = ("#F0D9B5", "#B58863"); sz = 60
        for r in range(8):
            for f in range(8):
                x, y = f*sz, (7-r)*sz
                draw.rectangle([x, y, x+sz, y+sz], fill=clr[(f+r)&1])
                pc = board.piece_at(chess.square(f, r))
                if pc:
                    k = ("w" if pc.color else "b") + pc.symbol().lower()
                    if k in self.imgs:
                        img.paste(self.imgs[k], (x, y), self.imgs[k])
        photo = ImageTk.PhotoImage(img); canvas.photo = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # ----- interacción usuario -----
    def _click(self, e):
        f, r = e.x // 60, 7 - e.y // 60
        sq = chess.square(f, r)
        if self.sel is None:
            self.sel = sq; return
        mv = chess.Move(self.sel, sq)
        if self.board.piece_type_at(self.sel) == chess.PAWN and chess.square_rank(sq) in (0, 7):
            promo = simpledialog.askstring("Promoción", "q,r,b,n:", initialvalue="q")
            mp = {"q": chess.QUEEN, "r": chess.ROOK,
                  "b": chess.BISHOP, "n": chess.KNIGHT}
            if promo in mp:
                mv = chess.Move(self.sel, sq, promotion=mp[promo])
        if mv in self.board.legal_moves:
            self.agent.record(self.board, mv)
            self.board.push(mv)
            self.agent.record(self.board)
            self._draw_board(); self.sel = None
            if self.board.is_game_over():
                self._end(); return
            threading.Thread(target=self._ia_move, daemon=True).start()
        else:
            messagebox.showwarning("Inválido", "Movimiento ilegal."); self.sel = None

    # IA mueve (thread)
    def _ia_move(self):
        mv = self.agent.select(self.board); self.queue.put(mv)

    # cola de acciones del hilo IA
    def _poll_queue(self):
        try:
            while True:
                mv = self.queue.get_nowait()
                self.agent.record(self.board, mv)
                self.board.push(mv)
                self.agent.record(self.board)
                self._draw_board()
                if self.board.is_game_over():
                    self._end()
        except queue.Empty:
            pass
        self.root.after(40, self._poll_queue)

    # mantenimiento cada minuto
    def _maint_loop(self):
        while True:
            time.sleep(60)
            self.mmc.decay()

    # bucle self‑play continuo
    def _selfplay_loop(self):
        a,b=Agent(self.mmc),Agent(self.mmc)
        board=chess.Board()
        opening_random_moves=4  # 2 jugadas por bando
        while True:
            if board.is_game_over():
                res=board.result()
                if res in("1-0","0-1"):
                    winner=a if res=="1-0" else b
                    loser=b if res=="1-0" else a
                    winner.finalize(1,0); loser.finalize(0,1)
                elif res=="1/2-1/2":
                    a.finalize(0,0); b.finalize(0,0)
                board=chess.Board(); a.prev_fen=b.prev_fen=None
                opening_random_moves=4
            ag=a if board.turn else b
            if opening_random_moves>0:
                mv=random.choice(list(board.legal_moves))
                opening_random_moves-=1
            else:
                mv=ag.select(board)
            ag.record(board,mv); board.push(mv); ag.record(board)
            self._draw_board(self.canvas,board)
            time.sleep(THINK_MS/1000)


    # ----- varios -----
    def _reset(self):
        self.board.reset(); self.agent.prev_fen = None; self._draw_board()

    def _reset_mmc(self):
        self.mmc.close(); self.mmc = MMC(); self.agent.mmc = self.mmc

    def _show_scores(self):
        tbl = sorted(((d["elo"], n) for n, d in self.scores.items()
                      if n != MACHINE_NAME), reverse=True)
        messagebox.showinfo("Posiciones",
                            "\n".join(f"{i+1}. {n} – {elo}" for i, (elo, n) in enumerate(tbl[:20]))
                            or "Vacío")

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
            if board.is_game_over():
                board = chess.Board(); w.prev_fen = b.prev_fen = None
            ag = w if board.turn else b
            mv = ag.select(board); ag.record(board, mv); board.push(mv); ag.record(board)
            self._draw_board(canvas, board); win.after(THINK_MS, step)

        step()

    # fin de partida
    def _end(self):
        res=self.board.result()
        sa=1 if res=="1-0" else 0 if res=="0-1" else .5
        winner=self.agent if sa==1 else None
        loser=self.agent if sa==0 else None
        if winner: winner.finalize(1,0)
        if loser:  loser.finalize(0,1)

        h,m=self.scores[self.player]["elo"], self.scores[MACHINE_NAME]["elo"]
        nh,nm=elo_update(h,m,sa)
        self.scores[self.player]["elo"],self.scores[MACHINE_NAME]["elo"]=round(nh),round(nm)
        save_scores(self.scores)

        game=chess.pgn.Game.from_board(self.board)
        self.mmc.store_game(str(game),res,self.player)

        msg="Ganaste" if sa==1 else "Empate" if sa==.5 else "Perdiste"
        if messagebox.askyesno("Fin",f"{msg}\n¿Revancha?"): self._reset()
        else: self.root.destroy()

    # ------------- resto igual que v9 (omitted) ---------------------------
if __name__=="__main__":
    ChessGUI()