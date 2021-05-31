import os
from itertools import cycle
import random
import pickle
import sys
import neat
import pygame
from pygame.locals import *

import visualize

from my_population import My_Population

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

pygame.init()

FPS = 1000000
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
FPSCLOCK = pygame.time.Clock()
STAT_FONT = pygame.font.SysFont("comicsans", 25)
THE_BEST_SCORE = 0
ALL_SCORES = list()

NUMBER_OF_GENERATION = 1
NUMBER_OF_REAL_RUN = 5000
CREATE_GRAPH = True
DRAW_THE_GAME = False
DRAW_LINES = True
TRAINING = False

NETS = []
BIRDS = []
BASEX = 0
BASESHIFT = 0
UPPERPIPES = None
LOWERPIPES = None
INDEX_OF_CURRENT_PIPE = 0

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

try:
    xrange
except NameError:
    xrange = range

gen = 0


class Bird:
    """
    Bird class representing the flappy bird
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velY = -9  # player's velocity along Y, default same as playerFlapped
        self.maxVelY = 10  # max vel along Y, max descend speed
        self.minVelY = -8  # min vel along Y, max ascend speed
        self.accY = 1  # players downward accleration
        self.rot = 45  # player's rotation
        self.velRot = 3  # angular speed
        self.rotThr = 20  # rotation threshold
        self.flapAcc = -9  # players speed on flapping
        self.flapped = False  # True when player flaps
        self.indexGen = cycle([0, 1, 2, 1])
        self.index = 0
        self.loopIter = 0


def main():
    global SCREEN
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    # SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
    # SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    # SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
    # SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    # SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_file.txt')

        if TRAINING:
            run(config_path)
            print('\nThe best score: \n', THE_BEST_SCORE)
        else:
            # realne hranie hry s vybranym genomom
            for x in range(NUMBER_OF_REAL_RUN):
                replay_genome(config_path)

            average_score = 0
            for score in ALL_SCORES:
                average_score += score
            average_score = average_score / len(ALL_SCORES)
            print('\nAverage score: \n', average_score)

            print('\nThe best score: \n', THE_BEST_SCORE)

            print('\n winner.pkl \n')

            # vykreslenie histogramu pri ukonceni realneho hrania
            n_bins = 20
            fig, axs = plt.subplots(1, 1)

            axs.hist(ALL_SCORES, bins=n_bins)
            plt.xlabel('Dosiahnuté skóre')
            plt.ylabel('Počet hier')

            plt.savefig('histogram')
            plt.show()


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                # make first flap sound and return values for mainGame
                # SOUNDS['wing'].play()
                return {
                    'playery': playery + playerShmVals['val'],
                    'basex': basex,
                    'playerIndexGen': playerIndexGen,
                }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)
        
        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(genomes, config):
    """ Hlavna cast hry """
    global gen, THE_BEST_SCORE, NETS, BIRDS, BASEX, BASESHIFT, UPPERPIPES, LOWERPIPES, INDEX_OF_CURRENT_PIPE
    gen += 1
    ge = []

    # vytvorenie neuronky pre kazdy genom a vytvaranie birdov
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        NETS.append(net)
        BIRDS.append(Bird(int(SCREENWIDTH * 0.2), int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)))
        ge.append(genome)

    score = 0
    BASESHIFT = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    UPPERPIPES = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    LOWERPIPES = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    while True and len(BIRDS) > 0:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                quit()

        for i in range(len(BIRDS)):
            birdMovement(i)

        # move pipes to left
        for uPipe, lPipe in zip(UPPERPIPES, LOWERPIPES):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

            for bird in BIRDS:
                # check for crash here
                crashTest = checkCrash({'x': bird.x, 'y': bird.y, 'index': bird.index},
                                       UPPERPIPES, LOWERPIPES)
                if crashTest[0]:
                    ge[BIRDS.index(bird)].fitness -= 1
                    NETS.pop(BIRDS.index(bird))
                    ge.pop(BIRDS.index(bird))
                    BIRDS.pop(BIRDS.index(bird))

        for x, bird in enumerate(BIRDS):
            # check for score
            # navysenie fitness, nastavenie indexu pre aktualnu pipu
            playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
            for pipe in UPPERPIPES:
                pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    ge[x].fitness += 1

                    # SOUNDS['point'].play()
                if playerMidPos > pipe['x'] + IMAGES['pipe'][0].get_width():
                    INDEX_OF_CURRENT_PIPE = 1

        # add new pipe when first pipe is about to touch left of screen
        # nastavenie indexu pre aktualnu pipu
        if len(UPPERPIPES) > 0 and 0 < UPPERPIPES[0]['x'] < 5:
            newPipe = getRandomPipe()
            UPPERPIPES.append(newPipe[0])
            LOWERPIPES.append(newPipe[1])
            UPPERPIPES.pop(0)
            LOWERPIPES.pop(0)
            INDEX_OF_CURRENT_PIPE = 0
            score += 1

            # SOUNDS['point'].play()

        # vykreslenie hry
        if DRAW_THE_GAME:
            # draw sprites
            SCREEN.blit(IMAGES['background'], (0, 0))

            for uPipe, lPipe in zip(UPPERPIPES, LOWERPIPES):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            SCREEN.blit(IMAGES['base'], (BASEX, BASEY))

            # score
            score_label = STAT_FONT.render("Score: " + str(score), True, (255, 255, 255))
            SCREEN.blit(score_label, (SCREENWIDTH - score_label.get_width() - 15, 10))

            # generations
            if TRAINING:
                score_label = STAT_FONT.render("Gens: " + str(gen - 1), True, (255, 255, 255))
                SCREEN.blit(score_label, (10, 10))
            else:
                score_label = STAT_FONT.render("Run: " + str(gen - 1), True, (255, 255, 255))
                SCREEN.blit(score_label, (10, 10))

            # alive
            score_label = STAT_FONT.render("Alive: " + str(len(BIRDS)), True, (255, 255, 255))
            SCREEN.blit(score_label, (10, 35))

            for bird in BIRDS:
                # Player rotation has a threshold
                visibleRot = bird.rotThr
                if bird.rot <= bird.rotThr:
                    visibleRot = bird.rot

                birdSurface = pygame.transform.rotate(IMAGES['player'][bird.index], visibleRot)
                SCREEN.blit(birdSurface, (bird.x, bird.y))

                # vykreslovanie ciar od birda k pipe
                if DRAW_LINES:
                    pygame.draw.line(SCREEN, (255, 0, 0),
                                     (bird.x + IMAGES['player'][0].get_width(),
                                      bird.y + IMAGES['player'][0].get_height()),
                                     (UPPERPIPES[INDEX_OF_CURRENT_PIPE]['x'] + IMAGES['pipe'][0].get_width() / 2,
                                      UPPERPIPES[INDEX_OF_CURRENT_PIPE]['y'] + IMAGES['pipe'][0].get_height()), 5)
                    pygame.draw.line(SCREEN, (255, 0, 0),
                                     (bird.x + IMAGES['player'][0].get_width(),
                                      bird.y + IMAGES['player'][0].get_height()),
                                     (LOWERPIPES[INDEX_OF_CURRENT_PIPE]['x'] + IMAGES['pipe'][0].get_width() / 2,
                                      LOWERPIPES[INDEX_OF_CURRENT_PIPE]['y']), 5)

        if DRAW_THE_GAME:
            pygame.display.update()
            FPSCLOCK.tick(FPS)

        # # break if score gets large enough
        # if score > 500:
        #     pickle.dump(nets[0], open("best.pickle", "wb"))
        #     break

    # ulozenie skore z aktualnej generacie a najlepsie skore
    print('Score: ' + str(score))
    ALL_SCORES.append(score)
    if score > THE_BEST_SCORE:
        THE_BEST_SCORE = score


def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx, playery))
        SCREEN.blit(IMAGES['gameover'], (50, 180))

        FPSCLOCK.tick(FPS)
        pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
        playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def birdMovement(i):
    """
    Metoda pre pohyb birda, vstup a vystup z neuronky
    :param i: predstavuje id birda v poli BIRDS
    :return: None
    """
    global NETS, BIRDS, BASEX, BASEY, BASESHIFT, UPPERPIPES, LOWERPIPES, INDEX_OF_CURRENT_PIPE

    # playerIndex basex change
    if (BIRDS[i].loopIter + 1) % 3 == 0:
        BIRDS[i].index = next(BIRDS[i].indexGen)
    BIRDS[i].loopIter = (BIRDS[i].loopIter + 1) % 30
    BASEX = -((-BASEX + 100) % BASESHIFT)

    # rotate the player
    if BIRDS[i].rot > -90:
        BIRDS[i].rot -= BIRDS[i].velRot

    # player's movement
    if BIRDS[i].velY < BIRDS[i].maxVelY and not BIRDS[i].flapped:
        BIRDS[i].velY += BIRDS[i].accY
    if BIRDS[i].flapped:
        BIRDS[i].flapped = False

        # more rotation to cover the threshold (calculated in visible rotation)
        BIRDS[i].rot = 45

    birdHeight = IMAGES['player'][BIRDS[i].index].get_height()
    BIRDS[i].y += min(BIRDS[i].velY, BASEY - BIRDS[i].y - birdHeight)

    # send bird location, top pipe location and bottom pipe location and determine from network whether to
    # jump or not
    output = NETS[i].activate(
        (BIRDS[i].y, abs(BIRDS[i].y - (UPPERPIPES[INDEX_OF_CURRENT_PIPE]['y'] + IMAGES['pipe'][0].get_height())),
         abs(BIRDS[i].y - (LOWERPIPES[INDEX_OF_CURRENT_PIPE]['y'])),
         (LOWERPIPES[INDEX_OF_CURRENT_PIPE]['x'] - BIRDS[i].x), BIRDS[i].velY))

    if output[0] > 25:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5
        # jump
        if BIRDS[i].y > -2 * IMAGES['player'][0].get_height():
            BIRDS[i].velY = BIRDS[i].flapAcc
            BIRDS[i].flapped = True
            # SOUNDS['wing'].play()


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = My_Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    list_of_bests = list()
    # Run for up to NUMBER_OF_GENERATION generations.
    winner, list_of_bests = p.run(mainGame, NUMBER_OF_GENERATION)

    index_of_best = ALL_SCORES.index(max(ALL_SCORES))
    pickle.dump(list_of_bests[index_of_best], open('winner.pkl', 'wb'))

    # vykreslenie grafu a modelu + ulozenie
    if CREATE_GRAPH:
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats)

        with open('winner.pkl', 'rb') as f:
            data = pickle.load(f)

        node_names = {-1: 'Bird', -2: 'Upper_pipe', -3: 'Lower_pipe', -4: 'Distance', -5: 'Velocity', 0: 'Jump'}
        visualize.draw_net(config, data, True, node_names=node_names)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


def replay_genome(config_path, genome_path="winner_012.pkl"):
    """
    Metoda pre spustenie realneho hrania natrenovaneho jedinca
     :param config_file: location of config file
     :param genome_path: file .pkl which represent the best genome
     :return: None
    """
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    print('GAME STARTED!')

    # Call game with only the loaded genome
    mainGame(genomes, config)


if __name__ == '__main__':
    main()
