#!/bin/bash

rm -rf ~/Downloads/test_movies/*
cp ~/Downloads/Torrents/completed/\[neoDESU\]\ SPY\ x\ FAMILY\ \[Season\ 1+2\]\ \[BD\ 1080p\ x265\ HEVC\ OPUS\ AAC\]\ \[Dual\ Audio\]/Season\ 1/SPY\ x\ FAMILY\ -\ S01E01.mkv ~/Downloads/test_movies/
cp ~/Downloads/Torrents/completed/\[neoDESU\]\ SPY\ x\ FAMILY\ \[Season\ 1+2\]\ \[BD\ 1080p\ x265\ HEVC\ OPUS\ AAC\]\ \[Dual\ Audio\]/Season\ 1/SPY\ x\ FAMILY\ -\ S01E02.mkv ~/Downloads/test_movies/
cp ~/Downloads/Torrents/completed/\[neoDESU\]\ SPY\ x\ FAMILY\ \[Season\ 1+2\]\ \[BD\ 1080p\ x265\ HEVC\ OPUS\ AAC\]\ \[Dual\ Audio\]/Season\ 1/SPY\ x\ FAMILY\ -\ S01E03.mkv ~/Downloads/test_movies/
python3 translate.py ~/Downloads/test_movies
