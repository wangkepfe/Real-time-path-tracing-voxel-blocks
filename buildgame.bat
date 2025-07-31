@echo off
cd build && cmake .. && cmake --build . --config Release --parallel && cd Release && voxelgame.exe