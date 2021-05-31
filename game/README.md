# Flappy Bird Bots - GA (AI)

Založené na [FlapPyBird](https://github.com/sourabhv/FlapPyBird) a [FlappyBirdBotsAI](https://github.com/martinglova/FlappyBirdBotsAI).

## Závislosti

Všetky potrebné závislosti, ktoré potrebujete nainštalovať sú v `requirements.txt`.

## Spustenie hry

Pre spustenie hry je treba spustiť `AI_GA_flappy_bird.py`.

## Súbory

#### The `AI_GA_flappy_bird.py` súbor

Súbor `AI_GA_flappy_bird.py` obsahuje samotnú hru. Máme tu rôzne premenné, a to: 
* `NUMBER_OF_GENERATION` ktorou určujeme počet generácií pri tréningu
* `NUMBER_OF_REAL_RUN` nastavujeme počet hier pri reálnom hrani jedného jedinca
* `CREATE_GRAPH` ak nastavíme na `True` tak na po zbehnutí tréningu uloži graf do `avg_fitness.svg`
* `DRAW_THE_GAME` hodnota `True` nám zabezpečí vykresľovanie hry
* `DRAW_LINES` ak nastavíme na `True` budeme vidieť čiary od birda k nasledujúcej pipe
* `TRAINING` ak bude nastavené na `True` tak sa bude vykonávať tréning, ak bude na `False` tak budeme hrať hru s jedným jedincom z `winner.pkl`  

#### The `config_file.txt` súbor

Súbor `config_file.txt` obsahuje základné parametre pre fungovanie NEAT-Python knižnice.

#### The `visualize.py` súbor

Pomocou tohto súboru vykresľujeme graf po zbehnutí všetkých generácií a taktiež vykresľujeme sieť najlepšieho jedinca.
