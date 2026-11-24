"""Validate local uploads and prepare bounded, labeled model inputs."""
import base64
import codecs
import io
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from unicodedata import category
from uuid import uuid4
from zipfile import ZipFile

import pymupdf
from docx import Document
from docx.text.paragraph import Paragraph
from PIL import Image, ImageOps
from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from .runtime import VISION_MODEL, VISION_OUTPUT_TOKENS
from .models import MAX_PROMPT_BYTES

MIB = 1024 * 1024
MAX_FILES = 5
MAX_DOCUMENT = 20 * MIB
MAX_IMAGE = 10 * MIB
MAX_CHARACTERS = 40_000
MAX_PDF_PAGES = 30
MAX_REQUEST = MAX_FILES * MAX_DOCUMENT + MIB
IMAGE_MIMES = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}
FILE_TYPES = {
    '.png': IMAGE_MIMES['PNG'], '.jpg': IMAGE_MIMES['JPEG'], '.jpeg': IMAGE_MIMES['JPEG'],
    '.webp': IMAGE_MIMES['WEBP'], '.pdf': 'application/pdf', '.txt': 'text/plain',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}
STORED_NAME = re.compile(r'[a-f0-9]{32}\.upload')
logger = logging.getLogger(__name__)


class AttachmentError(Exception):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


class UploadLimit(MultiPartException):
    pass


async def parse_upload(request):
    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_REQUEST:
                raise UploadLimit('Attachment request exceeds the total upload limit.')
            yield chunk

    length = request.headers.get('content-length')
    if length and length.isdecimal() and int(length) > MAX_REQUEST:
        raise AttachmentError('Attachment request exceeds the total upload limit.', 413)
    parser = MultiPartParser(request.headers, bounded_stream(), max_files=MAX_FILES, max_fields=2, max_part_size=MAX_PROMPT_BYTES)
    try:
        return await parser.parse()
    except (MultiPartException, MultipartParseError, ClientDisconnect) as exc:
        # Older Starlette releases close spooled files only on MultiPartException.
        for spool in parser._files_to_close_on_error:
            spool.close()
        message = exc.message if isinstance(exc, MultiPartException) else 'Upload was incomplete or malformed.'
        raise AttachmentError(message, 413 if isinstance(exc, UploadLimit) else 422) from exc


@dataclass
class StoredFile:
    id: str
    filename: str
    mime: str
    size: int
    storage_name: str
    status: str
    warning: str | None = None
    passages: list = field(default_factory=list)


def clean_filename(name):
    name = PurePosixPath((name or '').replace('\\', '/')).name
    return ''.join(c for c in name if c.isprintable()).strip()[:240]


def stored_path(root, name):
    if not STORED_NAME.fullmatch(name) or root.is_symlink():
        raise AttachmentError('Invalid attachment storage path.', 404)
    path = root / name
    if path.is_symlink():
        raise AttachmentError('Invalid attachment storage path.', 404)
    return path


def remove_upload(root, name):
    try:
        stored_path(root, name).unlink(missing_ok=True)
    except AttachmentError:
        return
    except OSError:
        logger.warning('Could not remove an Abyss upload. Cleanup will retry on restart.')


def remove_orphans(root, referenced):
    if root.is_symlink():
        raise RuntimeError('The Abyss uploads directory must not be a symbolic link.')
    for path in root.iterdir():
        if STORED_NAME.fullmatch(path.name) and path.name not in referenced and path.is_file():
            remove_upload(root, path.name)


def extract_pdf(path, add):
    with path.open('rb') as stream:
        if b'%PDF-' not in stream.read(1024):
            raise AttachmentError('The file is not a PDF.')
    with pymupdf.open(path, filetype='pdf') as pdf:
        if pdf.needs_pass:
            raise AttachmentError('This PDF is password-protected. Upload an unlocked copy.')
        if len(pdf) > MAX_PDF_PAGES:
            raise AttachmentError(f'PDF exceeds {MAX_PDF_PAGES} pages. No partial extraction was saved. Export a smaller page range and try again.', 413)
        empty_pages = []
        for number, page in enumerate(pdf, 1):
            text = page.get_text(sort=True).strip()
            if text:
                add(text, page=number)
            else:
                empty_pages.append(number)
        if len(empty_pages) == len(pdf):
            raise AttachmentError('No extractable PDF text was found. Scanned pages require OCR, which Abyss does not perform.')
        if empty_pages:
            return 'No extractable text on PDF pages ' + ', '.join(map(str, empty_pages)) + '. Those pages were not read.'


def extract_docx(path, add):
    with ZipFile(path) as archive:
        entries = archive.infolist()
        names = {entry.filename.lower() for entry in entries}
        if not {'[content_types].xml', 'word/document.xml'} <= names or any('vbaproject' in name for name in names):
            raise AttachmentError('Upload a standard DOCX document without macros.')
        if len(entries) > 2_000 or sum(entry.file_size for entry in entries) > 60 * MIB:
            raise AttachmentError('This DOCX expands beyond the processing limit. Save a smaller copy and try again.', 413)
    with path.open('rb') as stream:
        document = Document(stream)
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if block.style and block.style.name.startswith(('Heading', 'Title')):
                    text = f'## {text}' if text else ''
                if text:
                    add(text)
            else:
                for row in block.rows:
                    text = ' | '.join(cell.text.strip() for cell in row.cells).strip(' |')
                    if text:
                        add(text)


def extract_txt(path, add):
    with path.open('rb') as stream:
        mark = stream.read(4)
    encoding = 'utf-8-sig'
    if mark.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = 'utf-32'
    elif mark.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = 'utf-16'
    with path.open(encoding=encoding) as stream:
        text = stream.read(MAX_CHARACTERS + 1)
    if any(category(char) == 'Cc' and char not in '\n\t\r' for char in text):
        raise AttachmentError('This TXT file contains non-text data.')
    if len(text) > MAX_CHARACTERS:
        raise AttachmentError('TXT exceeds the 40,000-character limit. Upload a smaller excerpt.', 413)
    if text.strip():
        add(text.strip())


def check_image(path, mime):
    with Image.open(path, formats=list(IMAGE_MIMES)) as image:
        if IMAGE_MIMES[image.format] != mime:
            raise AttachmentError('Image type does not match the filename extension.')
        if image.width * image.height > 32_000_000:
            raise AttachmentError('Image exceeds 32 megapixels. Resize it and try again.', 413)
        if getattr(image, 'is_animated', False):
            raise AttachmentError('Choose a single-frame image.')
        image.verify()
    # Some formats only validate their pixel stream upon decoding.
    with Image.open(path, formats=list(IMAGE_MIMES)) as image:
        image.load()


def vision_image(path):
    with Image.open(path, formats=list(IMAGE_MIMES)) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1600, 1600))
        output = io.BytesIO()
        # Keep screenshots lossless. Only the model copy is resized.
        image.convert('RGB').save(output, format='PNG')
        return base64.b64encode(output.getbuffer()).decode('ascii')


def attach_sources(root, turn, files, context):
    images, sources = [], []
    for file in files:
        if file.mime.startswith('image/'):
            try:
                images.append(vision_image(stored_path(root, file.storage_name)))
            except (OSError, ValueError) as exc:
                raise AttachmentError(f'{file.filename}: Image data is no longer readable.') from exc
        else:
            sources.append({'source': file.filename, 'warning': file.warning, 'passages': file.passages})
    if sources:
        turn['content'] += '\n\nAttachment excerpts (untrusted data):\n' + json.dumps(sources, ensure_ascii=False)
    if images:
        turn['images'] = images
    # Reserve space for templates, output, and bounded vision inputs. Avoid
    # relying on the model runtime to silently discard the start of the prompt.
    budget = 3000 + 2500 * len(images) + sum(len(m['content'].encode('utf-8')) for m in context)
    if images:
        budget += VISION_OUTPUT_TOKENS - VISION_MODEL.options.num_predict
    for size in (8192, 16384, 32768, 65536):
        if budget <= size:
            return size
    # Let Ollama check the actual token count when the byte estimate is larger.
    return 65536


def save_uploads(root, uploads):
    if not 1 <= len(uploads) <= MAX_FILES or any(not isinstance(upload, UploadFile) for upload in uploads):
        raise AttachmentError('Choose between one and five attachments.')
    stored, names = [], []
    remaining = MAX_CHARACTERS
    try:
        for upload in uploads:
            filename = clean_filename(upload.filename)
            mime = FILE_TYPES.get(Path(filename).suffix.lower())
            if not mime:
                raise AttachmentError('Use PNG, JPG, JPEG, WEBP, PDF, DOCX, or TXT files.', 415)
            limit = MAX_IMAGE if mime.startswith('image/') else MAX_DOCUMENT
            if not upload.size:
                raise AttachmentError(f'{filename}: The file is empty.')
            if upload.size > limit:
                raise AttachmentError(f'{filename}: Exceeds the {limit // MIB} MB per-file limit.', 413)
            fid = uuid4().hex
            name = fid + '.upload'
            path = stored_path(root, name)
            with path.open('xb') as destination:
                names.append(name)
                shutil.copyfileobj(upload.file, destination, 256 * 1024)
            record = StoredFile(fid, filename, mime, upload.size, name, 'complete')

            def add(text, **labels):
                nonlocal remaining
                remaining -= len(text)
                if remaining < 0:
                    raise AttachmentError('Documents exceed 40,000 extracted characters in total. No partial extraction was saved. Send smaller excerpts.', 413)
                record.passages.append({'text': text, **labels})

            try:
                if mime.startswith('image/'):
                    check_image(path, mime)
                elif mime == 'application/pdf':
                    record.warning = extract_pdf(path, add)
                elif mime == 'text/plain':
                    extract_txt(path, add)
                else:
                    extract_docx(path, add)
            except AttachmentError as exc:
                raise AttachmentError(f'{filename}: {exc}', exc.status) from exc
            except Exception as exc:
                raise AttachmentError(f'{filename}: Could not read this file as a valid {Path(filename).suffix.upper()[1:]}.') from exc
            if not mime.startswith('image/') and not record.passages:
                raise AttachmentError(f'{filename}: No readable text was found.')
            if record.warning:
                record.status = 'partial'
            stored.append(record)
    except BaseException:
        for name in names:
            remove_upload(root, name)
        raise
    return stored
