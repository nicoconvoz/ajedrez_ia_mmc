# MMC Chess Learner v21

### Motor de ajedrez auto‑evolutivo con Memoria de Mapa Conceptual (MMC)

MMC Chess es un laboratorio vivo para experimentar con **IA evolutiva** aplicada al ajedrez. El programa combina una interfaz gráfica sencilla (Tkinter), un grafo de memoria persistente en **MySQL**, y un agente que aprende por refuerzo mediante partidas *self‑play* y análisis de bases PGN.

El proyecto incluye dos componentes principales:

| Archivo | Propósito |
|---------|-----------|
| `mmc_chess_v21_async.py` | Aplicación GUI + Agente MMC + Self‑play asíncrono |
| `pgn_loader.py` | Importador masivo de partidas PGN a la base `mmc_chess` |

## Características clave

* **MMC persistente** – Los nodos (FEN) y aristas (movimientos) se refuerzan con recompensas/penalizaciones, guardando metadatos como captura, escape, victoria o derrota.
* **Aprendizaje continuo** – Bucle IA‑vs‑IA en segundo plano; el grafo se depura con *decay* y poda automática.
* **Heurística mixta** – El agente combina aperturas almacenadas, pensamiento lateral y un MCTS ligero con tablas de valores de pieza.
* **Sistema Elo local** – Actualiza el ranking de cada usuario y de la máquina tras cada partida.
* **Visualización de clusters** – Dibuja comunidades de posiciones con NetworkX + Louvain.
* **Importación de PGN** – `pgn_loader.py` procesa miles de partidas, incrementando el peso de cada transición y registrando estadísticas de captura, win/loss.

## Requisitos

```bash
Python >= 3.10
MySQL 8  (o compatible)
Tkinter (incluido en la mayoría de distribuciones)

pip install -r requirements.txt
# contenidos mínimos:
# python‑chess mysql‑connector‑python pillow matplotlib networkx tqdm
```

## Instalación rápida

```bash
# 1. Clonar el repositorio
$ git clone https://github.com/<usuario>/mmc‑chess.git
$ cd mmc‑chess

# 2. Crear y activar un entorno virtual (opcional pero recomendado)
$ python -m venv .venv
$ source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
$ pip install -r requirements.txt

# 4. Crear la base de datos MySQL
mysql> CREATE DATABASE mmc_chess DEFAULT CHARSET utf8mb4;

# 5. Copiar `db_config.json` y completar credenciales:
{
  "user": "root",
  "password": "",
  "host": "127.0.0.1",
  "database": "mmc_chess"
}
```

## Ejecución

### GUI interactiva

```bash
$ python mmc_chess_v21_async.py
```

* Registro/Login con usuario y contraseña.
* Elige *Jugar contra la IA* o *Ver IA vs IA*.
* Durante la partida puedes:
  * Ver los *Clusters MMC* desde *Analyze*.
  * Reiniciar partida o memoria.

### Carga masiva de PGN

```bash
$ python pgn_loader.py ./pgn‑mega‑database/
```

El script recorre carpetas, detecta `.pgn`, valida el esquema y hace *commit* cada 100 partidas.

## Esquema de base de datos

```text
nodes  (fen PK, last_update DOUBLE)
edges  (src PK, dst PK,
        weight DOUBLE,
        capture TINYINT,
        escape  TINYINT,
        win INT,
        loss INT)
games  (id PK, pgn MEDIUMTEXT, result VARCHAR(7), player VARCHAR(64), ts TIMESTAMP)
```

La migración es automática; si las columnas `win` o `loss` faltan, se añaden en caliente.

## Arquitectura interna

```
┌─────────┐    moves      ┌──────────────┐
│  GUI    │──────────────▶│  Agent (MCTS)│
└─────────┘               └──────┬───────┘
      ▲                           │ update(weight,flags)
      │ fen                       ▼
┌─────┴────┐  choose(fen)   ┌────────────┐
│   MMC    │───────────────▶│   MySQL    │
└──────────┘   (explore)    └────────────┘
```

1. **Agent** pide a la MMC el mejor destino para la posición actual (FEN).
2. Se aplica una política de exploración + creatividad.
3. El usuario mueve; el *reward* refuerza la arista en la BD.
4. Un *flush worker* en segundo plano agrupa INSERT/UPDATE asíncronos para reducir latencia.

## Contribución

¡Serán bienvenidas *issues* y *pull requests*!

1. Crea una rama descriptiva.
2. Sigue *PEP 8* y añade docstrings donde falten.
3. Incluye pruebas unitarias cuando sea razonable.
4. Firma tu *commit* si es posible.

## Licencia

Este proyecto se publica bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más detalles.

## Créditos

* Basado en `python‑chess` de Niklas Fiekas.
* Inspirado en la propuesta de **Memoria de Mapa Conceptual (MMC)** de la comunidad.

---

> _«El ajedrez es el gimnasio de la mente.»_ – Blaise Pascal


# MMC Chess Learner & PGN Bulk Loader

**MMC Chess** is an experimental chess engine + self-training GUI that stores game knowledge in a *Memoria de Mapa Conceptual* (MMC) graph inside MySQL.
The companion **PGN Loader** script lets you import thousands of games to pre-seed the MMC database.

---

## Features

| Component                    | Highlights                                                                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`mmc_chess_v21_async.py`** | • Tk-based GUI (Human vs AI, AI vs AI) • Login / ELO rating board • Lightweight MCTS + MMC priors • Self-play background training • Community detection visualiser (NetworkX)  |
| **`pgn_loader.py`**          | • Recursive folder import • Batched commits (MySQL) • Adds *win / loss / capture* statistics to MMC edges • Progress bars (tqdm)                                               |

---

## Quick-start

```bash
# 1) clone
git clone https://github.com/<your-user>/mmc_chess.git
cd mmc_chess

# 2) create virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) install requirements
pip install -r requirements.txt

# 4) configure MySQL
cp db_config.sample.json db_config.json
# edit user / password / host if needed

# 5) launch the GUI
python mmc_chess_v21_async.py

# Optional: bulk-import grand-master games
python pgn_loader.py /path/to/PGN/folder
```

---

## Requirements

* **Python 3.9 +** (tested on 3.12)
* **MySQL 8.x** (or MariaDB 10.5+) with a user allowed to create databases/tables.

### Python packages

| Package                  | Used for                       |
| ------------------------ | ------------------------------ |
| `mysql-connector-python` | DB access                      |
| `python-chess`           | board representation & PGN IO  |
| `Pillow`                 | piece images (PNG)             |
| `matplotlib`             | cluster plot                   |
| `networkx`               | Louvain community detection    |
| `tqdm`                   | progress bars in PGN loader    |

A ready-made **`requirements.txt`** is included:

```
mysql-connector-python
python-chess
pillow
matplotlib
networkx
tqdm
```

---

## Database configuration

Create a `db_config.json` (same folder as the scripts):

```json
{
  "user"    : "root",
  "password": "",
  "host"    : "127.0.0.1",
  "database": "mmc_chess",
  "raise_on_warnings": false
}
```

*At first run the engine auto-creates the `mmc_chess` schema and any missing columns.*&#x20;

---

## Folder layout

```
mmc_chess/
├── images/              # 60×60 PNG sprites named wp.png, bp.png, wq.png…
├── mmc_chess_v21_async.py
├── pgn_loader.py
├── requirements.txt
├── db_config.sample.json
└── README.md
```

> **Piece images**
> Supply standard chess piece PNGs (or your own set) in `images/`.
> Without them the GUI still works but shows empty squares.

---

## Usage

### Play against the AI

```bash
python mmc_chess_v21_async.py
```

1. **Login / register** with any username + password (local `scoreboard.json`).
2. Choose *Jugar contra la IA*.
3. The agent learns after every move; ELO updates automatically at game end.

### Watch self-play

*Menu →* **Play → Ver IA vs IA…**
or start a dedicated window from the welcome screen.

### Visualise MMC clusters

*Menu →* **Analyze → Clusters MMC**
(Opens a matplotlib window with Louvain communities.)

### Resetting memory

*Menu →* **Play → Reset memoria IA** – drops the current MySQL connection and recreates a fresh MMC.

---

## Bulk-import PGN archives

```bash
python pgn_loader.py /dir/with/pgns      # recursive
python pgn_loader.py lichess_2024.pgn    # single file
```

* Commits in batches of **100** games (`BATCH_SIZE` constant).
* Updates `nodes`, `edges`, and `games` tables; adds `weight`, `win`, `loss`, `capture` stats.

---

## Tips & Troubleshooting

| Problem                                                              | Fix                                                                              |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `mysql.connector.errors.ProgrammingError: 1054 Unknown column 'win'` | Run the current version – it performs live *ALTER TABLE* if fields are missing.  |
| GUI shows blank board                                                | Ensure `images/*.png` exist (white/black pieces).                                |
| `ModuleNotFoundError`                                                | Re-install dependencies inside the venv: `pip install -r requirements.txt`.      |
| Connection refused                                                   | Check MySQL is running and credentials in `db_config.json` are correct.          |

---

## Contributing

Pull requests are welcome! Feel free to:

* Add stronger evaluation heuristics or NN back-ends
* Improve the PGN loader (parallel import, escape flag…)
* Share larger PGN datasets or better piece graphics

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

Enjoy teaching an MMC-powered engine to play creative chess!
