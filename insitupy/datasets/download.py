import logging
import os
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_url(
    url: str,
    out_dir: Union[str, os.PathLike, Path] = ".",
    file_name: Optional[str] = None,
    chunk_size: int = 65536,
    overwrite: bool = False
    ) -> None:
    """
    Downloads a file from the specified URL and saves it to the given output directory.

    Code adapted from: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    Args:
        url (str): The URL of the file to be downloaded.
        out_dir (Union[str, os.PathLike, Path], optional): The output directory where the downloaded file will be saved.
            Default is the current directory (".").
        file_name (str, optional): The name of the downloaded file. If not provided, the function will use the name
            from the URL. Default is None.
        chunk_size (int, optional): The size of the chunks in bytes to download the file. Default is 65536 bytes (64 KB).
            Larger values can improve download speed for large files.
        overwrite (bool, optional): If True, the function will download the file even if it already exists in the
            output directory, overwriting the existing file. If False and the file exists, the function will skip
            the download. Default is False.

    Returns:
        None: This function does not return any value. The downloaded file is saved in the specified output directory.
    """
    # create output directory if necessary
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # check which file name to use
    if file_name is None:
        # Use the full filename from URL
        outfile = out_dir / Path(url).name
    else:
        # If file_name already has an extension, use it as-is
        # Otherwise, get suffix from URL
        if '.' in file_name:
            outfile = out_dir / file_name
        else:
            # Get suffix from URL (handle multiple dots like .ome.tif or .tar.gz)
            url_name = Path(url).name
            # Find the first dot to separate name from extension(s)
            if '.' in url_name:
                suffix = url_name[url_name.index('.'):]  # Everything after first dot
            else:
                suffix = ''
            outfile = out_dir / (file_name + suffix)

    if outfile.exists():
        if not overwrite:
            logger.info(f"File {outfile} exists already. Download is skipped. To force download set `overwrite=True`.")
            return
        else:
            pass


    # request content from URL
    # Use a session for better connection handling and add headers similar to browsers
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

    # Configure session for better performance
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=3,
        pool_block=False
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    resp = session.get(url, stream=True, headers=headers, timeout=(10, 30))
    resp.raise_for_status()  # Raise an error for bad status codes
    total = int(resp.headers.get('content-length', 0))

    # write to file with buffered writing
    with open(str(outfile), 'wb', buffering=1024*1024) as file, tqdm(
        desc=str(outfile.name),
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            if data:  # filter out keep-alive new chunks
                size = file.write(data)
                bar.update(size)