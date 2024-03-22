import urllib
from tqdm import tqdm

# Modified from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file(url, file_path, chunk_size=1024):
    req = urllib.request.urlopen(url)
    with open(file_path, 'wb') as f, tqdm(
        desc=f'Downloading {file_path}', total=req.length, unit='iB', 
        unit_scale=True, unit_divisor=1024
    ) as bar:
        for _ in range(req.length // chunk_size + 1):
            chunk = req.read(chunk_size)
            if not chunk: break
            size = f.write(chunk)
            bar.update(size)