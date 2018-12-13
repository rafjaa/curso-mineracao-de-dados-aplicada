import json

from flask import Flask
from flask import request
from flask import render_template


app = Flask(__name__)


@app.route('/', methods=['GET'])
def api():
    if request.method == 'GET':
        return render_template('base.html')


@app.route('/analisa_noticia', methods=['POST'])
def post():
    txt = request.form.get('txt')
    print(txt)

    ''' 
        Atividade: obter a probabilidade da notícia ser sobre esporte:

        1) Preprocessar a notícia recebida (txt)
        2) Vetorizar com o mesmo TfidfVectorizer usado no dataset de treino
        3) Usar a matriz tf-idf de uma linha resultante como entrada do modelo
           Prever com predict_proba(X)[:,1]
        4) Jogar o resultado na variável "proba" (abaixo)

    '''
    proba = 0.95

    return json.dumps({'proba': proba})


if __name__ == '__main__':
    app.run(debug=True)