
#!/usr/bin/env python3
"""PGN Bulk Loader for MMC Chess MySQL Database

Loads multiple PGN files into the `mmc_chess` MySQL database.
Populates `games`, `nodes`, and `edges` tables, updating weights
and capture/escape statistics.

Usage
-----
python pgn_loader.py /path/to/folder/with/pgns
python pgn_loader.py file1.pgn file2.pgn

Requirements
------------
pip install python-chess mysql-connector-python tqdm
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import chess
import chess.pgn
import mysql.connector
from mysql.connector import errorcode
from tqdm import tqdm

# --- Configuration -------------------------------------------------------

DB_CONFIG = {
    "user":     "root",
    "password": "",
    "host":     "127.0.0.1",
    "database": "mmc_chess",
    "raise_on_warnings": False,
    "autocommit": False,
}

BATCH_SIZE = 100   # commit every N games

# ------------------------------------------------------------------------

def get_files(paths):
    """Resolve command‑line paths into a list of PGN file paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.pgn"), recursive=True))
        else:
            files.append(p)
    return sorted(set(files))

def connect_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            sys.exit("Invalid MySQL credentials")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            sys.exit("Database does not exist")
        else:
            sys.exit(err)

def ensure_schema(cur):
    # Ensure tables exist (executed once)
    cur.execute("""CREATE TABLE IF NOT EXISTS nodes(
                        fen VARCHAR(120) PRIMARY KEY,
                        last_update DOUBLE)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS edges(
                        src VARCHAR(120), dst VARCHAR(120),
                        weight DOUBLE DEFAULT 0,
                        capture TINYINT DEFAULT 0,
                        escape  TINYINT DEFAULT 0,
                        win     INT DEFAULT 0,
                        loss    INT DEFAULT 0,
                        PRIMARY KEY(src,dst),
                        INDEX(src), INDEX(dst))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS games(
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        pgn MEDIUMTEXT,
                        result VARCHAR(7),
                        player VARCHAR(64),
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""ALTER TABLE edges
                    ADD COLUMN IF NOT EXISTS win INT DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS loss INT DEFAULT 0""")

def insert_game(cur, pgn_text, result, player):
    sql = """INSERT INTO games(pgn, result, player) VALUES (%s, %s, %s)"""
    cur.execute(sql, (pgn_text, result, player))

def upsert_node(cur, fen, ts):
    sql = ("INSERT INTO nodes (fen, last_update) VALUES (%s, %s) "
           "ON DUPLICATE KEY UPDATE last_update = VALUES(last_update)")
    cur.execute(sql, (fen, ts))

def upsert_edge(cur, src, dst, capture, win_delta, loss_delta):
    # escape flag unimplemented (placeholder)
    sql = ("INSERT INTO edges (src, dst, weight, capture, win, loss) "
           "VALUES (%s, %s, 1, %s, %s, %s) "
           "ON DUPLICATE KEY UPDATE weight = weight + 1, "
           "capture = capture + VALUES(capture), "
           "win = win + VALUES(win), "
           "loss = loss + VALUES(loss)")
    cur.execute(sql, (src, dst, capture, win_delta, loss_delta))

def process_game(cur, game):
    result = game.headers.get("Result", "*")
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    pgn_io = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_text = game.accept(pgn_io)

    insert_game(cur, pgn_text, result, white)

    board = game.board()
    fen_src = board.fen()
    ts = datetime.utcnow().timestamp()
    upsert_node(cur, fen_src, ts)

    for move in game.mainline_moves():
        board.push(move)
        fen_dst = board.fen()
        upsert_node(cur, fen_dst, ts)

        capture_flag = int(board.is_capture(move))
        # Determine outcome for the edge perspective (White POV)
        if result == "1-0":
            win_delta, loss_delta = 1, 0
        elif result == "0-1":
            win_delta, loss_delta = 0, 1
        else:
            win_delta = loss_delta = 0

        upsert_edge(cur, fen_src, fen_dst, capture_flag, win_delta, loss_delta)
        fen_src = fen_dst

# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bulk import PGN files into mmc_chess DB")
    parser.add_argument("paths", nargs="+", help="PGN files or directories")
    args = parser.parse_args()

    files = get_files(args.paths)
    if not files:
        print("No PGN files found.")
        return

    conn = connect_db()
    cur = conn.cursor()
    ensure_schema(cur)

    try:
        with tqdm(total=len(files), desc="Files", unit="file") as file_bar:
            game_bar = None
            batch_count = 0

            for fpath in files:
                file_bar.set_postfix(file=os.path.basename(fpath))
                file_bar.refresh()

                with open(fpath, encoding="utf-8", errors="ignore") as fp:
                    game_bar = tqdm(desc=os.path.basename(fpath),
                                    unit="game", leave=False)
                    while True:
                        game = chess.pgn.read_game(fp)
                        if game is None:
                            break
                        process_game(cur, game)
                        batch_count += 1
                        game_bar.update(1)

                        if batch_count >= BATCH_SIZE:
                            conn.commit()
                            batch_count = 0
                    game_bar.close()
                file_bar.update(1)

            if batch_count:
                conn.commit()
        print("Import finished successfully.")
    except KeyboardInterrupt:
        print("Interrupted! Rolling back...")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
