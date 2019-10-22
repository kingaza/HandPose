from flask import Flask

import pygame


idx_song = 0
songs = ['./Songs/fly.mp3', './Songs/hero.mp3', './Songs/love.mp3']

pygame.mixer.init()
pygame.mixer.music.load(songs[idx_song])


app = Flask(__name__)

@app.route('/')
def index():
    return 'I am a Server.'

@app.route('/ptab/move', methods=['GET', 'POST'])
def move():
    return 'Move PTab.'    

@app.route('/ptab/stop', methods=['GET', 'POST'])
def stop():
    return 'Stop PTab.'        

@app.route('/music/play', methods=['GET', 'POST'])
def music_play():
    pygame.mixer.music.play()
    return 'Play a song.'    

@app.route('/music/stop', methods=['GET', 'POST'])
def music_stop():
    pygame.mixer.music.stop()
    return 'Stop playing song.'    

@app.route('/music/last', methods=['GET', 'POST'])
def music_last():
    global idx_song
    idx_song = (idx_song - 1) % len(songs)
    pygame.mixer.music.load(songs[idx_song])
    pygame.mixer.music.play()
    return 'Play last song.'    

@app.route('/music/next', methods=['GET', 'POST'])
def music_next():
    global idx_song
    idx_song = (idx_song + 1) % len(songs)
    pygame.mixer.music.load(songs[idx_song])
    pygame.mixer.music.play()    
    return 'Play next song.'                

if __name__ == '__main__':
    app.debug = True # 设置调试模式，生产模式的时候要关掉debug
    app.run()