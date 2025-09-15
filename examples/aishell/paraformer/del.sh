#!/bin/bash

# 目标目录
TARGET_DIR="/mnt/maxiangnan/mrx/m_cif/FunASR/examples/aishell/paraformer/pic_val"
# TARGET_DIR="/mnt/maxiangnan/mrx/m_cif/FunASR/examples/aishell/paraformer/test_dir"
# 保留文件数 
KEEP=10

# 进入目标目录
cd "$TARGET_DIR" || exit 1

# 检查文件数量
count=$(ls -1 | wc -l)

if [ "$count" -gt "$KEEP" ]; then
    echo "$(date): 当前文件数 $count，执行清理..."
    
    # 按修改时间倒序排列，保留最新的KEEP个文件
    # 使用ls -t获取按时间排序列表（最新的在前）
    # tail -n +11 从第11个文件开始获取（需要删除的旧文件）
    files_to_delete=$(ls -t | tail -n +$(($KEEP+1)))
    
    # 删除旧文件
    rm -f $files_to_delete
    
    echo "已删除文件："
    echo "$files_to_delete"
    echo "----------------------------------"
fi
