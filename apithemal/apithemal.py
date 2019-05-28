from flask import Flask, request, send_from_directory, render_template
import tempfile
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', code_text=None, code_hteml=None, prediction=None, error=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return index()

    code = '\n'.join(map(str.strip, request.form['code'].encode('utf-8').strip().split('\n')))
    try:
        prediction = get_prediction_of_code(code)
        error = None
    except ValueError as v:
        prediction = None
        error = v.args[0]

    return render_template(
        'index.html',
        code_text=code,
        code_html=code.replace('\n', '<br>'),
        prediction=prediction,
        error=error,
    )


def get_prediction_of_code(code):
    _, fname = tempfile.mkstemp()
    assemble_success = (
        intel_compile(code, fname) or
        att_compile(code, fname) or
        nasm_compile(code, fname)
    )

    if not assemble_success:
        if os.path.exists(fname):
            os.unlink(fname)
        raise ValueError('Could not assemble code')

    try:
        return float(subprocess.check_output([
            'python',
            '/home/ithemal/ithemal/learning/pytorch/ithemal/predict.py',
             '--model', '/home/ithemal/ithemal/learning/pytorch/saved/predictor.dump',
            '--model-data', '/home/ithemal/ithemal/learning/pytorch/saved/trained.mdl',
            '--files', fname
        ]).strip())
    except:
        if os.path.exists(fname):
            os.unlink(fname)
        raise ValueError('Ithemal failed to predict timing of code')


def intel_compile(code, output):
    p = subprocess.Popen(['as', '-o', output], stdin=subprocess.PIPE)
    c = '''
        .text
        .global main
        main:
        .intel_syntax noprefix

        mov ebx, 111
        .byte 0x64, 0x67, 0x90

        {}

        mov ebx, 222
        .byte 0x64, 0x67, 0x90
    '''.format(code)
    p.communicate(c)
    p.wait()
    return p.returncode == 0

def att_compile(code, output):
    p = subprocess.Popen(['as', '-o', output], stdin=subprocess.PIPE)
    c = '''
        .text
        .global main
        main:

        movq $111, %ebx
        .byte 0x64, 0x67, 0x90

        {}

        mov $222, %ebx
        .byte 0x64, 0x67, 0x90
    '''.format(code)
    p.communicate(c)
    p.wait()
    return p.returncode == 0

def nasm_compile(code, output):
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'w+') as f:
        f.write('''
	SECTION .text
        global main
        main:

        mov ebx, 111
        db 0x64, 0x67, 0x90

        {}

        mov ebx, 222
        db 0x64, 0x67, 0x90
        '''.format(code))

    p = subprocess.Popen(['nasm', '-o', output, tmp.name], stdin=subprocess.PIPE)
    p.wait()
    tmp.close()
    return p.returncode == 0
