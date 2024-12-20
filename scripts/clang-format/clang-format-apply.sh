#!/bin/bash

#
# This is apllying clang-format
# by doing a filtering step and then
# applying clang-format-and-fix-macros.sh
#

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
TARGET_BRANCH=$1

# check that we are in a clean state in order to prevent accidental
# changes
if [ ! -z "$(git status --untracked-files=no  --porcelain)" ]; then
  echo "Script must be applied on a clean git state"
  exit 1
fi


# Retrieve list of files that were changed in source branch
# with respect to master (target branch)
filelist=
function get_filelist() {
    # 用 files_all 存储git追踪的所有文件
    local files_all=`git diff -U0 --no-color $TARGET_BRANCH --name-only`

    # 创建一个新的空变量来存放过滤后的结果
    local files_filtered=""

    # 读取 files_all 变量中的每一行
    while read -r line; do
        # 过滤掉一些不希望被自动格式化的文件
        if [[ ! $line == examples/cpp/utils/CLI11.hpp && \
              ! $line == third_party/* && \
              ! $line == HIE-DNN/* && \
              ! $line == span-attention/* && \
              ! $line == tests/cpp/model/stresstest/input.h && \
              ! $line == csrc/core/kernel/cuda/xformer_mha/kernel_forward.h && \
              ! $line == csrc/interface/dlpack.h && \
              ! $line == csrc/utility/concurrentqueue.h && \
              ! $line == csrc/utility/cnpy.* ]]; then
            if [ -L "$line" ]; then
                : # 过滤掉软链接
            else
                files_filtered+="$line"$'\n'
            fi
        fi
    done <<< "$files_all"

    # 移除最后一个换行符
    files_filtered="${files_filtered%$'\n'}"

    # files_filtered 变量中包含了所有过滤后的文件
    # echo "$files_filtered"
    filelist=${files_filtered}
}

get_filelist

# function to check if C++ file (based on suffix)
# can probably be done much shorter
function checkCPP(){
    if [[ $1 == *.cc ]];then
	return 0
    elif [[ $1 == *.cpp ]];then
	return 0
    elif [[ $1 == *.cxx ]];then
	return 0
    elif [[ $1 == *.C ]];then
	return 0
    elif [[ $1 == *.c++ ]];then
	return 0
    elif [[ $1 == *.c ]];then
	return 0
    elif [[ $1 == *.CPP ]];then
	return 0
	# header files
    elif [[ $1 == *.h ]];then
	return 0
    elif [[ $1 == *.hpp ]];then
	return 0
    elif [[ $1 == *.hh ]];then
	return 0
    elif [[ $1 == *.icc ]];then
	return 0
    # cuda
    elif [[ $1 == *.cu ]];then
	return 0
    elif [[ $1 == *.cuh ]];then
	return 0
    fi
    return 1
}

echo
echo "Checking formatting using the following clang-format version:"
clang-format --version
echo

# check list of files
for f in $filelist; do
    if checkCPP $f; then
	echo "CHECKING MATCHING FILE ${f}"
	# apply the clang-format script
	clang-format -i ${f}
    fi
done

# check if something was modified
notcorrectlist=`git status --porcelain | grep '^ M' | cut -c4-`
# if nothing changed ok
if [[ -z $notcorrectlist ]]; then
  # send a negative message to git
  echo "Excellent. **VERY GOOD FORMATTING!** :thumbsup:"
  exit 0;
else
  echo "The following files have clang-format problems (showing patches)";
  git diff $notcorrectlist
fi

exit 1



