from tqdm import tqdm # Librería para barras de progreso
import pandas as pd
import otelo
import mcts

partidas = 30
ganadas_negras = 0
ganadas_blancas = 0

df_estados_partidas = pd.DataFrame()

def crear_fila(tablero, turno, ganador):
    fila = tablero.copy()
    fila["turno"] = turno
    fila["ganador"] = ganador
    return fila

for i in tqdm(range(partidas)):

    turno = 1

    tablero = otelo.crear_tablero()

    #otelo.mostrar_tablero(tablero)

    partida_actual = []

    while True:

        if not otelo.jugador_puede_mover(tablero, 1) and not otelo.jugador_puede_mover(tablero, 2):
            ganador, blancas, negras = otelo.ganador(tablero)
            if ganador == 1:
                ganadas_negras += 1
                print("Las negras ganan la partida")
            elif ganador == 2:
                ganadas_blancas += 1
                print("Las blancas ganan la partida")
            print("Las negras han conseguido: ", negras)
            print("Las blancas han conseguido: ", blancas)
            # Todas las posiciones de la partida se etiquetan con el ganador FINAL
            # (antes se guardaba quién iba ganando en cada jugada, que no es lo que la red debe aprender).
            filas = [crear_fila(estado, t, ganador) for estado, t in partida_actual]
            df_estados_partidas = pd.concat([df_estados_partidas, pd.DataFrame(filas)], ignore_index=True)
            break


        if not otelo.jugador_puede_mover(tablero, turno):
            turno = 3 - turno
            continue

        if turno == 1:
            movimiento = mcts.mcts_uct(tablero, turno, iteraciones=50)
        elif turno == 2:
            movimiento = mcts.mcts_uct(tablero, turno, iteraciones=10, red_neuronal=True)

        otelo.poner_ficha(tablero, movimiento[0], movimiento[1], turno)

        # La etiqueta se pone al terminar la partida: aquí solo se guarda el estado.
        partida_actual.append((tablero.copy(), turno))
        
        turno = 3 - turno

#df_estados_partidas.to_csv("partidas_ia_vs_ia.csv", index=False)
print("Partidas ganadas por las negras: ", ganadas_negras)
print("Partidas ganadas por las blancas: ", ganadas_blancas)