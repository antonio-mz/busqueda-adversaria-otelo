# Othello: Monte Carlo Tree Search with a learned value network

An Othello (Reversi) engine in Python. A Monte Carlo Tree Search (MCTS) agent plays games against itself, a neural network learns from those games to evaluate board positions, and the network can replace the random playouts inside the search.

The question this repository answers: **does the learned network make the search play better?** Measured carefully, **it does not.** The rest of this README explains how that was measured and why.

## Background

This started as a university project (Artificial Intelligence course, University of Seville, 2025), built by Manuel Jesús Sánchez García and Antonio Montero López. The original report is in `documentacion.tex` and `PROYECTO_IA___Aprendiendo_a_jugar_a_Otelo.pdf`.

This fork (2026) revisits the project, fixes its bugs and replaces the original evaluation with a measured one. **The report's claim that the network-guided agent won about 70-75% of its games does not hold.** It was measured with the bugs listed below.

## What was fixed

1. **MCTS move selection and reward sign** (`mcts.py`). The UCT selection never advanced its index, so it always returned the first (randomly shuffled) child, and rewards were stored from the wrong player's point of view, so each node preferred the move best for its opponent. The final move is now the most visited child. Against a random player, the original MCTS won 10 of 20 games; after the fix it won 20 of 20.
2. **Training labels.** Every position is now labelled with the final winner of its game, not with whoever was ahead at that moment.
3. **Data leakage in the evaluation** (`modelo.py`). Train and test were split by individual positions, so near-identical positions from the same game ended up on both sides and the test score was inflated. Train, validation and test are now split by whole games.
4. **Turn encoding.** The network was trained with the player who had just moved, but the search fed it the player about to move. Small effect, now consistent.
5. **Inference speed.** Keras `predict()` cost about 46 ms per position, which made equal-budget comparisons impractical. The search now runs the same forward pass in NumPy (about 75 µs per position, same output to within 1e-6).

## Results

### The network as an evaluator

750 MCTS-vs-MCTS games (1,000 iterations per move), split by game into 480 for training, 120 for validation and 150 for test. The training set is augmented with the 8 symmetries of the board (rotations and reflections), which do not change the outcome.

| Model | Test MSE | Predicts the winner |
|---|---|---|
| Always predict a draw (0) | 0.993 | — |
| Always pick the side that won most training games | — | 51.0% |
| **Network** | **0.866** | **62.3%** |

Accuracy grows as the game advances: 51% in the opening (0-23 discs), 60% in the middlegame (24-43) and 74.5% in the endgame (44-64). That is expected: early in the game the outcome is close to a coin flip.

### The network inside the search

100 games per row, colours alternating, 95% Wilson confidence intervals:

| Match | Network side wins |
|---|---|
| Network MCTS (1,000 it.) vs plain MCTS (1,000 it.) | **3** (1 draw), 95% CI 1.0-8.5% |
| Network MCTS (200 it.) vs plain MCTS (1,000 it.) | **1**, 95% CI 0.2-5.4% |

Against a player that moves at random (20 games each):

| Player | Wins vs random |
|---|---|
| Plain MCTS | 20/20 |
| Network MCTS | 17/20 |
| Network MCTS with the network's sign inverted | 7/20 |
| Network alone, one move ahead | 9/20 |

### Why the network fails

- **The sign is right.** Inverting it makes the agent worse than random, so this is not a perspective bug.
- **The network knows who is ahead, but cannot rank moves.** Looking one move ahead it plays no better than random. Choosing a move means telling apart sibling positions that differ by one move, and the network's values for them differ by a median of 0.32, small next to the UCT exploration term (0.7-1.9).
- **Random playouts carry more information.** Each one plays the game to the end and sees the real result.

### Ideas not pursued

- Encode each square as two inputs (black, white) instead of the ordinal 0/1/2, which treats a white disc as "twice" a black one.
- A convolutional network that sees the board as a grid.
- Mix the network's value with short playouts, as AlphaGo did.
- Retune the exploration constant to the scale of the network's values.

## How to run

Python 3.12. Tested with numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, tensorflow 2.21.0, keras 3.15.1, h5py 3.14.0 and tqdm 4.70.1.

```bash
python main.py                                         # play against the engine
python generar_datos.py --partidas 150 --procesos 7    # self-play data (resumable)
python modelo.py                                       # train and evaluate the network
python experimento.py --iter-red 1000 --iter-mcts 1000 --partidas 100
python experimento.py --resumen                        # summarize saved results
```

Datasets: `partidas_ia_vs_ia.csv` (games 0-149) and `partidas_extra.csv` (games 150-749). Match results: `experimento_resultados.csv`.

## Files

| File | Purpose |
|---|---|
| `otelo.py` | Board, legal moves and rules |
| `mcts.py` | MCTS with random playouts or network evaluation |
| `generar_datos.py` | Parallel self-play data generation |
| `modelo.py` | Training and evaluation of the value network |
| `experimento.py` | Network MCTS vs plain MCTS matches |
| `main.py` | Human vs engine |
| `ia_vs_ia.py` | Original match script, kept for reference |
