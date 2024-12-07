#!/bin/bash

#
# This is running a clang-format test
# by doing a filtering step and then analysing
# the result of applying clang-format-and-fix-macros.sh
#

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
DEFAULT_TARGET_BRANCH="origin/dashscope_dev"
TARGET_BRANCH="${1:-$DEFAULT_TARGET_BRANCH}"
$SCRIPTPATH/clang-format-apply.sh $TARGET_BRANCH
res=$?

if [ $res -eq 0 ] ; then 
   # cleanup changes in git
   git reset HEAD --hard
fi

exit $res
