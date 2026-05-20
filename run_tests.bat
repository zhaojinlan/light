@echo off
REM 运行测试并将结果输出到 test_results.txt（覆盖写入）
python -m pytest test/ > test_results.txt 2>&1
type test_results.txt
