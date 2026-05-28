import os
import zipfile

GDRIVE_FILE_ID = '1lGosJiylVgNld1Wq5olBBG2DwfNaJYrE'

DATASET_FILES = {
    'amazon': ['trn_mat.pkl', 'val_mat.pkl', 'tst_mat.pkl',
               'usr_emb_np.pkl', 'itm_emb_np.pkl', 'usr_prf.pkl', 'itm_prf.pkl'],
    'yelp':   ['trn_mat.pkl', 'val_mat.pkl', 'tst_mat.pkl',
               'usr_emb_np.pkl', 'itm_emb_np.pkl', 'usr_prf.pkl', 'itm_prf.pkl'],
}

# load_data/ is one level below the project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dataset_complete(name):
    for fname in DATASET_FILES[name]:
        if not os.path.exists(os.path.join(_ROOT, 'data', name, fname)):
            return False
    return True


def ensure_datasets():
    missing = [ds for ds in DATASET_FILES if not _dataset_complete(ds)]
    if not missing:
        return

    print(f'[Data] Missing datasets: {missing}. Downloading from Google Drive...')

    try:
        import gdown
    except ImportError:
        raise ImportError(
            'gdown is required for automatic dataset download.\n'
            'Install it with: pip install gdown'
        )

    zip_path = os.path.join(_ROOT, '_datasets_tmp.zip')

    gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)

    print('[Data] Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(_ROOT)

    os.remove(zip_path)
    print('[Data] Datasets ready.')
