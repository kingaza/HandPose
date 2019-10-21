from flask import Flask

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

if __name__ == '__main__':
    app.debug = True # 设置调试模式，生产模式的时候要关掉debug
    app.run()