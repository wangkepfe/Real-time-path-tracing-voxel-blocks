@echo off
cd build && cmake .. && cmake --build . --config Release --parallel && cd Release && voxelgame_offline.exe --test-canonical --frames 64 && start offline_render_0063.png