# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for SentinelBonsai.
Build with:  pyinstaller SentinelBonsai.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Collect all needed packages
whisper_datas, whisper_binaries, whisper_hiddenimports = collect_all('whisper')
pyttsx3_datas, pyttsx3_binaries, pyttsx3_hiddenimports = collect_all('pyttsx3')

all_datas = whisper_datas + pyttsx3_datas
all_binaries = whisper_binaries + pyttsx3_binaries
all_hiddenimports = (
    whisper_hiddenimports
    + pyttsx3_hiddenimports
    + [
        'pyttsx3',
        'pyttsx3.drivers',
        'pyttsx3.drivers.sapi5',
        'speech_recognition',
        'numpy',
        'torch',
        'tqdm',
        'numba',
        'regex',
    ]
)

a = Analysis(
    ['launcher.py', 'SentinelBonsai.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SentinelBonsai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,     # Keep console for voice assistant output
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SentinelBonsai',
)
