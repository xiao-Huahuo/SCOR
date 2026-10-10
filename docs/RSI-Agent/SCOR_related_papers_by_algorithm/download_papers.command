#!/bin/bash
set -u
cd "$(dirname "$0")"
exec python3 -u download_papers.py
