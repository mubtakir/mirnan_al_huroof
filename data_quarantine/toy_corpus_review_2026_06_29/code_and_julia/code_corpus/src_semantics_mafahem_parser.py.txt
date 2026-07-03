import os
import glob

_MAFAHEM_DICT = {}
_LOADED = False

def load_mafahem(directory=None):
    global _LOADED, _MAFAHEM_DICT
    if _LOADED:
        return _MAFAHEM_DICT
        
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mafahem')
        
    if not os.path.exists(directory):
        return _MAFAHEM_DICT
        
    try:
        files = glob.glob(os.path.join(directory, '*.txt'))
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('←')
                    if len(parts) == 2:
                        symbol = parts[0].strip()
                        concept = parts[1].strip()
                        if symbol and concept:
                            _MAFAHEM_DICT[symbol] = concept
                            
        _LOADED = True
    except Exception as e:
        print(f"Error loading mafahem: {e}")
        
    return _MAFAHEM_DICT

def get_concept(symbol):
    if not _LOADED:
        load_mafahem()
    return _MAFAHEM_DICT.get(symbol, None)

def is_symbol(word):
    if not _LOADED:
        load_mafahem()
    return word in _MAFAHEM_DICT
