import fitz
from pathlib import Path
from glob import glob

base_path = 'sources'
pdf_files = glob(base_path + '/*.pdf')
