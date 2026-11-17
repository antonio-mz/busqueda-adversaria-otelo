"""Mide si la red neuronal hace jugar mejor al MCTS.

Enfrenta un MCTS que evalúa las hojas con la red contra un MCTS con simulaciones
aleatorias. Los colores se alternan (partidas pares: la red juega con negras), así
que la ventaja de mover primero se reparte. Cada partida se añade al CSV en cuanto
termina; relanzar el mismo comando se salta las que ya están.

Uso:
    python experimento.py --iter-red 1000 --iter-mcts 1000 --partidas 100
    python experimento.py --resumen
"""

import argparse
import math
import os
import random
import time
from multiprocessing import Pool

import pandas as pd

import mcts
import otelo

SALIDA = "experimento_resultados.csv"
COLUMNAS = ["iter_red", "iter_mcts", "partida", "color_red", "ganador", "fichas_red", "fichas_mcts"]


def jugar(args):
    """Play one game network-MCTS vs random-rollout MCTS and return its result row."""
    iter_red, iter_mcts, num_partida, semilla = args
    random.seed(semilla)
    color_red = 1 if num_partida % 2 == 0 else 2

    tablero = otelo.crear_tablero()
    turno = 1
    while True:
        if not otelo.jugador_puede_mover(tablero, 1) and not otelo.jugador_puede_mover(tablero, 2):
            ganador, blancas, negras = otelo.ganador(tablero)
            break
        if not otelo.jugador_puede_mover(tablero, turno):
            turno = 3 - turno
            continue
        if turno == color_red:
            movimiento = mcts.mcts_uct(tablero, turno, iteraciones=iter_red, red_neuronal=True)
        else:
            movimiento = mcts.mcts_uct(tablero, turno, iteraciones=iter_mcts)
        otelo.poner_ficha(tablero, movimiento[0], movimiento[1], turno)
        turno = 3 - turno

    fichas = {1: negras, 2: blancas}
    return {
        "iter_red": iter_red, "iter_mcts": iter_mcts, "partida": num_partida,
        "color_red": color_red, "ganador": ganador,
        "fichas_red": fichas[color_red], "fichas_mcts": fichas[3 - color_red],
    }


def intervalo_wilson(exitos, n, z=1.96):
    """95 % Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = exitos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - margen, centro + margen


def resumir(df):
    for (iter_red, iter_mcts), g in df.groupby(["iter_red", "iter_mcts"]):
        gana_red = (g["ganador"] == g["color_red"]).sum()
        empates = (g["ganador"] == 0).sum()
        n = len(g)
        bajo, alto = intervalo_wilson(gana_red, n)
        print(f"\nRed {iter_red} it. contra MCTS {iter_mcts} it. ({n} partidas)")
        print(f"  Gana la red: {gana_red} ({gana_red / n:.1%}, IC 95 % {bajo:.1%}-{alto:.1%}), "
              f"empates: {empates}, pierde: {n - gana_red - empates}")
        for color, nombre in ((1, "negras"), (2, "blancas")):
            c = g[g["color_red"] == color]
            if len(c):
                print(f"  Con {nombre}: gana {(c['ganador'] == color).sum()} de {len(c)}")
        print(f"  Fichas medias: red {g['fichas_red'].mean():.1f}, MCTS {g['fichas_mcts'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iter-red", type=int, default=1000)
    parser.add_argument("--iter-mcts", type=int, default=1000)
    parser.add_argument("--partidas", type=int, default=100)
    parser.add_argument("--procesos", type=int, default=7)
    parser.add_argument("--semilla", type=int, default=20261117)
    parser.add_argument("--resumen", action="store_true", help="only summarize saved results")
    args = parser.parse_args()

    if args.resumen:
        resumir(pd.read_csv(SALIDA))
        return

    hechas = set()
    if os.path.exists(SALIDA):
        previo = pd.read_csv(SALIDA)
        mismo = previo[(previo["iter_red"] == args.iter_red) & (previo["iter_mcts"] == args.iter_mcts)]
        hechas = set(mismo["partida"])
    tareas = [(args.iter_red, args.iter_mcts, n, args.semilla + n)
              for n in range(args.partidas) if n not in hechas]

    # Workers get the weights as numpy arrays, so they never import TensorFlow.
    pesos = mcts.extraer_pesos()
    inicio = time.time()
    with Pool(args.procesos, initializer=mcts.usar_pesos, initargs=(pesos,)) as pool:
        for i, fila in enumerate(pool.imap_unordered(jugar, tareas), start=1):
            pd.DataFrame([fila], columns=COLUMNAS).to_csv(
                SALIDA, mode="a", header=not os.path.exists(SALIDA), index=False)
            minutos = (time.time() - inicio) / 60
            resultado = "red" if fila["ganador"] == fila["color_red"] else ("empate" if fila["ganador"] == 0 else "mcts")
            print(f"[{i}/{len(tareas)}] gana {resultado} ({fila['fichas_red']}-{fila['fichas_mcts']}) | "
                  f"{minutos:.1f} min, quedan ~{minutos / i * (len(tareas) - i):.0f} min", flush=True)

    df = pd.read_csv(SALIDA)
    resumir(df[(df["iter_red"] == args.iter_red) & (df["iter_mcts"] == args.iter_mcts)])


if __name__ == "__main__":
    main()
