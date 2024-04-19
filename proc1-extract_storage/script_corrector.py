from flask import Flask, render_template, request, redirect, session, jsonify
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import re
# from underthesea import text_normalize
import py_vncorenlp

vncorenlp_segmentor = py_vncorenlp.VnCoreNLP(annotators=['wseg'], 
                                            #  max_heap_size='-Xmx4g',
                                             save_dir='/workspace/nlplab/kienvt/scada-tokenize-server/vncorenlp')
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction", device=0, max_new_tokens=512)
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
app = Flask(__name__)
BSZ = 8
# MAX_LENGTH = 100
# OVERLAPSE = 0.5

class MyDataset(Dataset):
    def __init__(self, document): 
        self._document = []
        for doc in document:
            _doc = re.sub(r'\.(\s)?\.', '.', doc)
            i = 0
            _doc = _doc.replace('\t', '. ').replace('\n', '. ')
            _doc = vncorenlp_segmentor.word_segment(_doc)
            # _doc = [d for d in _doc.split('.') if len(d.strip()) > 0]
            while i + 1 < len(_doc):
                _doc[i] = _doc[i].strip().replace('_', ' ')
                if 0 < len(_doc[i]) <= 7: 
                    _doc[i+1] = f'{_doc[i]} {_doc[i+1].strip()}'
                    _doc.pop(i)
                elif len(_doc[i]) == 0:
                    _doc.pop(i)
                else:
                    i += 1
            #end-while
            self._document += _doc
    
    def __len__(self):
        return len(self._document)
    
    def __getitem__(self, index):
        return self._document[index]

@app.route('/health', methods=['GET'])
def check_health():
    return jsonify({'status': 'OK'}), 200


@app.route('/', methods=['POST'])
def correct():
    data = request.json
    dset = MyDataset(data['document'])
    document_ = []
    with torch.no_grad():
        for out in tqdm(corrector(dset, batch_size=BSZ, max_length=None), total=len(dset)):
            document_ += list([d["generated_text"] for d in out])
    del dset
    return jsonify({'generated_text' : document_}), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=9299, debug=False)