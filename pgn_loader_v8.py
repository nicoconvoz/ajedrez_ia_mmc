
#!/usr/bin/env python3
"""
PGN Bulk Loader for MMC Chess – **v7‑canon (23‑May‑2025)**
─────────────────────────────────────────────────────────
Cambios respecto a v6:
• Usa FEN *canónico* (solo los 4 primeros campos) para que coincida con
  la clave primaria que emplea *mmc_chess_v43*.
• Etiqueta `win`/`loss` **por color que mueve** en cada jugada,
  no global por partida.
• Mantiene cálculo de características y compatibilidad con la tabla
  `features`.

Uso
----
    python pgn_loader_v7.py carpeta_con_pgns
    python pgn_loader_v7.py file1.pgn file2.pgn
"""
from __future__ import annotations
import argparse, glob, os, sys
from datetime import datetime
import chess, chess.pgn
import mysql.connector
from mysql.connector import errorcode
from tqdm import tqdm

# ───────────────── CONFIGURACIÓN ───────────────────────────────────────
DB_CONFIG = {
    "user":     "root",
    "password": "",
    "host":     "127.0.0.1",
    "database": "mmc_chess",
    "raise_on_warnings": False,
    "autocommit": False,
}

BATCH_SIZE = 100  # commit cada N partidas
# ───────────────────────────────────────────────────────────────────────

# ---------- FEN canónico ----------------------------------------------
def canon(board: chess.Board) -> str:
    """Devuelve los 4 campos estables del FEN."""
    return " ".join(board.fen().split()[:4])

# ---------- funciones de características ------------------------------
def extract_features(board: chess.Board):
    """Devuelve un diccionario con métricas simples de la posición."""
    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    mat_diff = sum(piece_values[p.piece_type] if p.color else -piece_values[p.piece_type]
                   for p in board.piece_map().values())
    atk_w = def_w = atk_b = def_b = 0
    for sq, pc in board.piece_map().items():
        attacked = board.attackers(not pc.color, sq)
        defended = board.attackers(pc.color, sq)
        if pc.color:
            atk_w += bool(attacked)
            def_w += bool(defended)
        else:
            atk_b += bool(attacked)
            def_b += bool(defended)
    ks_w = int(board.is_check() and board.turn)
    ks_b = int(board.is_check() and not board.turn)
    mob_w = len(list(board.legal_moves)) if board.turn else 0
    mob_b = len(list(board.legal_moves)) if not board.turn else 0
    return dict(mat_diff=mat_diff, attackers_w=atk_w, attackers_b=atk_b,
                defenders_w=def_w, defenders_b=def_b,
                king_safety_w=ks_w, king_safety_b=ks_b,
                mobility_w=mob_w, mobility_b=mob_b)

def upsert_features(cur, fen: str, feat: dict, ts: float):
    cols = ("mat_diff","attackers_w","attackers_b","defenders_w","defenders_b",
            "king_safety_w","king_safety_b","mobility_w","mobility_b")
    placeholders = ",".join(["%s"]*(len(cols)+2))
    sql = (f"INSERT INTO features (fen,{','.join(cols)},last_update) "
           f"VALUES({placeholders}) "
           f"ON DUPLICATE KEY UPDATE " + ", ".join([f"{c}=VALUES({c})" for c in cols]) +
           ", last_update=VALUES(last_update)")
    cur.execute(sql, (fen, *(feat[c] for c in cols), ts))

# ---------------------------------------------------------------------
def get_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.pgn"), recursive=True))
        else:
            files.append(p)
    return sorted(set(files))

def connect_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            sys.exit("Credenciales MySQL inválidas")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            sys.exit("La base de datos no existe")
        else:
            sys.exit(err)

def ensure_schema(cur):
    # NODES -------------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS nodes(
                        fen VARCHAR(120) PRIMARY KEY,
                        last_update DOUBLE)""")
    # EDGES -------------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS edges(
                        src VARCHAR(120), dst VARCHAR(120),
                        weight DOUBLE DEFAULT 0,
                        capture TINYINT DEFAULT 0,
                        `escape` TINYINT DEFAULT 0,
                        `check`  TINYINT DEFAULT 0,
                        mate      TINYINT DEFAULT 0,
                        draw_push TINYINT DEFAULT 0,
                        win INT DEFAULT 0,
                        loss INT DEFAULT 0,
                        PRIMARY KEY(src,dst), INDEX(src), INDEX(dst))""")
    # migrar columnas faltantes
    cur.execute("SHOW COLUMNS FROM edges"); cols = {r[0] for r in cur.fetchall()}
    want = {"escape":"TINYINT","check":"TINYINT","mate":"TINYINT",
            "draw_push":"TINYINT","win":"INT","loss":"INT"}
    for col, typ in want.items():
        if col not in cols:
            cur.execute(f"ALTER TABLE edges ADD COLUMN {col} {typ} DEFAULT 0")
    # FEATURES ----------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS features(
                        fen VARCHAR(120) PRIMARY KEY,
                        mat_diff SMALLINT,
                        attackers_w TINYINT, attackers_b TINYINT,
                        defenders_w TINYINT, defenders_b TINYINT,
                        king_safety_w TINYINT, king_safety_b TINYINT,
                        mobility_w TINYINT, mobility_b TINYINT,
                        last_update DOUBLE)""")
    # GAMES -------------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS games(
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        pgn MEDIUMTEXT,
                        result VARCHAR(7),
                        player VARCHAR(64),
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

# ---------------- helpers DB ----------------
def insert_game(cur, pgn_text, result, player):
    cur.execute("INSERT INTO games(pgn,result,player) VALUES(%s,%s,%s)",
                (pgn_text, result, player))

def upsert_node(cur, fen: str, ts: float):
    cur.execute("INSERT INTO nodes(fen,last_update) VALUES(%s,%s) "
                "ON DUPLICATE KEY UPDATE last_update=VALUES(last_update)",
                (fen, ts))

def upsert_edge(cur, src: str, dst: str, capture: int,
                check: int, mate: int, draw_push: int,
                win_delta: int, loss_delta: int):
    cur.execute(
        """INSERT INTO edges(src,dst,weight,capture,`check`,mate,draw_push,win,loss)
               VALUES(%s,%s,1,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE
                   weight=weight+1,
                   capture=capture+VALUES(capture),
                   `check`=`check`+VALUES(`check`),
                   mate=mate+VALUES(mate),
                   draw_push=draw_push+VALUES(draw_push),
                   win=win+VALUES(win),
                   loss=loss+VALUES(loss)""", 
        (src, dst, capture, check, mate, draw_push, win_delta, loss_delta))

# --------------- PROCESO PARTIDA ----------------
def process_game(cur, game):
    result = game.headers.get("Result", "*")
    pgn_text = game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True))
    insert_game(cur, pgn_text, result, game.headers.get("White", ""))

    board = game.board()
    ts = datetime.utcnow().timestamp()
    fen_prev = canon(board)
    upsert_node(cur, fen_prev, ts)
    upsert_features(cur, fen_prev, extract_features(board), ts)

    winner_is_white = (result == "1-0")
    decisive = result in ("1-0", "0-1")

    for mv in game.mainline_moves():
        color_to_move = board.turn  # antes del push
        is_capture = board.is_capture(mv)
        board.push(mv)
        fen_curr = canon(board)

        upsert_node(cur, fen_curr, ts)
        upsert_features(cur, fen_curr, extract_features(board), ts)

        check_flag = int(board.is_check())
        mate_flag  = int(board.is_checkmate())
        draw_flag  = int(board.can_claim_draw())

        if decisive:
            win_delta  = int(color_to_move == winner_is_white)
            loss_delta = int(color_to_move != winner_is_white)
        else:
            win_delta = loss_delta = 0

        upsert_edge(cur, fen_prev, fen_curr, int(is_capture),
                    check_flag, mate_flag, draw_flag,
                    win_delta, loss_delta)
        fen_prev = fen_curr

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Importa masivamente archivos PGN a la BD mmc_chess (FEN canónico)")
    ap.add_argument("paths", nargs="+", help="Archivos o carpetas con PGN")
    args = ap.parse_args()

    files = get_files(args.paths)
    if not files:
        print("No se encontraron PGN."); return

    conn = connect_db(); cur = conn.cursor()
    ensure_schema(cur)

    try:
        with tqdm(total=len(files), desc="Files", unit="file") as file_bar:
            batch = 0
            for fpath in files:
                file_bar.set_postfix(file=os.path.basename(fpath)); file_bar.refresh()
                with open(fpath, encoding="utf-8", errors="ignore") as fp:
                    with tqdm(desc=os.path.basename(fpath), unit="game", leave=False) as game_bar:
                        while True:
                            game = chess.pgn.read_game(fp)
                            if game is None: break
                            process_game(cur, game)
                            batch += 1; game_bar.update(1)
                            if batch >= BATCH_SIZE:
                                conn.commit(); batch = 0
                file_bar.update(1)
            if batch: conn.commit()
        print("Importación finalizada con éxito.")
    except KeyboardInterrupt:
        print("¡Interrumpido! rollback…"); conn.rollback()
    finally:
        cur.close(); conn.close()

# ---------------------------------------------------------------------
# ------------------- v8 PATCH (25‑may‑2025) ---------------------------
# Sincroniza el cargador PGN con mmc_chess_v55:
# 1) Añade columna `last_update` y `escape` si faltan.
# 2) Rellena y actualiza `last_update` en cada upsert_edge.
# 3) Calcula `escape_flag` (rey estaba en jaque antes del movimiento).
# ---------------------------------------------------------------------

# --- Ajustar ensure_schema ------------------------------------------
def ensure_schema(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS nodes(
                        fen VARCHAR(120) PRIMARY KEY,
                        last_update DOUBLE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS edges(
                        src VARCHAR(120), dst VARCHAR(120),
                        weight DOUBLE DEFAULT 0,
                        capture TINYINT DEFAULT 0,
                        `escape` TINYINT DEFAULT 0,
                        `check`  TINYINT DEFAULT 0,
                        mate      TINYINT DEFAULT 0,
                        draw_push TINYINT DEFAULT 0,
                        win INT DEFAULT 0,
                        loss INT DEFAULT 0,
                        last_update DOUBLE,
                        PRIMARY KEY(src,dst),
                        INDEX(src), INDEX(dst))""")
    # migrar columnas faltantes
    cur.execute("SHOW COLUMNS FROM edges")
    cols = {r[0] for r in cur.fetchall()}
    want = { "escape":"TINYINT", "check":"TINYINT", "mate":"TINYINT",
             "draw_push":"TINYINT", "win":"INT", "loss":"INT",
             "last_update":"DOUBLE"}
    for col, typ in want.items():
        if col not in cols:
            cur.execute(f"ALTER TABLE edges ADD COLUMN {col} {typ} DEFAULT 0")
    # FEATURES ---------------------------------------------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS features(
                        fen VARCHAR(120) PRIMARY KEY,
                        mat_diff SMALLINT,
                        attackers_w TINYINT, attackers_b TINYINT,
                        defenders_w TINYINT, defenders_b TINYINT,
                        king_safety_w TINYINT, king_safety_b TINYINT,
                        mobility_w TINYINT, mobility_b TINYINT,
                        last_update DOUBLE)""")
    # GAMES ------------------------------------------------------------
    cur.execute("""CREATE TABLE IF NOT EXISTS games(
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        pgn MEDIUMTEXT,
                        result VARCHAR(7),
                        player VARCHAR(64),
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

# --- upsert_edge con escape y last_update ----------------------------
def upsert_edge(cur, src: str, dst: str, capture: int,
                escape_flag: int, check: int, mate: int, draw_push: int,
                win_delta: int, loss_delta: int, ts: float):
    cur.execute(
        """INSERT INTO edges(src,dst,weight,capture,`escape`,`check`,mate,draw_push,win,loss,last_update)
               VALUES(%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE
                   weight=weight+1,
                   capture=capture+VALUES(capture),
                   `escape`=`escape`+VALUES(`escape`),
                   `check`=`check`+VALUES(`check`),
                   mate=mate+VALUES(mate),
                   draw_push=draw_push+VALUES(draw_push),
                   win=win+VALUES(win),
                   loss=loss+VALUES(loss),
                   last_update=VALUES(last_update)""", 
        (src, dst, capture, escape_flag, check, mate, draw_push,
         win_delta, loss_delta, ts))

# ------------------ Proceso partida patch ---------------------------
def process_game(cur, game):
    result = game.headers.get("Result", "*")
    pgn_text = game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True))
    insert_game(cur, pgn_text, result, game.headers.get("White", ""))

    board = game.board()
    ts = datetime.utcnow().timestamp()
    fen_prev = canon(board)
    upsert_node(cur, fen_prev, ts)
    upsert_features(cur, fen_prev, extract_features(board), ts)

    winner_is_white = (result == "1-0")
    decisive = result in ("1-0", "0-1")

    for mv in game.mainline_moves():
        color_to_move = board.turn  # antes del push
        in_check_before = board.is_check()
        is_capture = board.is_capture(mv)

        board.push(mv)
        fen_curr = canon(board)

        upsert_node(cur, fen_curr, ts)
        upsert_features(cur, fen_curr, extract_features(board), ts)

        check_flag = int(board.is_check())
        mate_flag  = int(board.is_checkmate())
        draw_flag  = int(board.can_claim_draw())

        if decisive:
            win_delta  = int(color_to_move == winner_is_white)
            loss_delta = int(color_to_move != winner_is_white)
        else:
            win_delta = loss_delta = 0

        upsert_edge(cur, fen_prev, fen_curr, int(is_capture),
                    int(in_check_before), check_flag, mate_flag, draw_flag,
                    win_delta, loss_delta, ts)
        fen_prev = fen_curr

if __name__ == "__main__":
    main()
