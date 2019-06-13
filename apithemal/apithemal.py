from flask import Flask, request, send_from_directory, render_template
import tempfile
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', code_text=None, code_hteml=None, prediction=None, error=None)

def strip_comment(line):
    if ';' in line:
        return line[:line.index(';')]
    return line

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return index()

    code = '\n'.join(map(strip_comment, map(str.strip, request.form['code'].encode('utf-8').strip().split('\n'))))

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
        error=(error and error.replace('\n', '<br>')),
    )


def get_prediction_of_code(code):
    with open('/home/ithemal/apithemal_log', 'a') as f:
        f.write('='*80+'\n'+code+'\n'+'='*80+'\n')

    _, fname = tempfile.mkstemp()
    success, as_intel_output = intel_compile(code, fname)
    if not success:
        success, as_att_output = att_compile(code, fname)
    if not success:
        success, nasm_output = nasm_compile(code, fname)

    if not success:
        if os.path.exists(fname):
            os.unlink(fname)
        raise ValueError('Could not assemble code.\nAssembler outputs:\n\n{}'.format('\n\n'.join([
            'as (Intel syntax): {}'.format(as_intel_output[1]),
            'as (AT&T syntax): {}'.format(as_att_output[1]),
            'nasm: {}'.format(nasm_output[1]),
        ])))

    try:
        return '{:.3f}'.format(float(subprocess.check_output([
            'python',
            '/home/ithemal/ithemal/learning/pytorch/ithemal/predict.py',
             '--model', '/home/ithemal/ithemal/learning/pytorch/saved/predictor.dump',
            '--model-data', '/home/ithemal/ithemal/learning/pytorch/saved/trained.mdl',
            '--files', fname
        ]).strip()) / 100)
    except:
        if os.path.exists(fname):
            os.unlink(fname)
        raise ValueError('Ithemal failed to predict timing of code')


def intel_compile(code, output):
    p = subprocess.Popen(['as', '-o', output], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    output = p.communicate(c)
    p.wait()
    return p.returncode == 0, output

def att_compile(code, output):
    p = subprocess.Popen(['as', '-o', output], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    output = p.communicate(c)
    p.wait()
    return p.returncode == 0, output

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

    p = subprocess.Popen(['nasm', '-o', output, tmp.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate()
    p.wait()
    tmp.close()
    return p.returncode == 0, output
