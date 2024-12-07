Changelog
=========

1.5.5 (2021-05-20)
-----

* Fix #172: Added 'size_t' to typecasts detected by CheckCStyleCast
* Fixed wrong CLI help text: Each filter needs + or -
* Fix #164: add elif as an exception for CheckSpacingForFunctionCall()
* Fix google#346: --root option not working on windows due to slashes in path

1.5.4 (2020-08-18)
-----

* Fix google#166, Allow space before C++11 attributes

1.5.3 (2020-07-20)
-----

* Fix #156: sed/gsed output parameter rejected
* Fix #156: sed/gsed output without other stdout information
* improvements to regression tests

1.5.2 (2020-06-24)
-----

* Fix #83, output formats "sed" and "gsed" to auto-fix some issues
* Fix #92, new category "build/namespaces_headers" for unnamed namespaces in header file
* Sort list of files before processing
* Fix #144 Falso positive for indent when using QT macros "signals" and "slots"
* Fix #76 Parsing of class decorators that also use digits
* Fix #139 Add message "Relative paths like . and .. are not allowed"

1.5.1 (2020-06-05)
-----

* Revert #43 behavior change for include order from 1.5.0, and hide it behind command-line-flag `--includeorder=standardcfirst`.
  It turns out there is no easy objective way to tell c system headers from certain c++ library headers, and Google cpplint intentionally classifies some C++ header includes as C system header for simplicity.
* Libraries considered as C system headers using --includeorder=standardcfirst now also includes linux-specifc headers (glibc-devel, glibc-kernheaders, linux-libc-dev).


1.5.0 (2020-05-31)
-----

* Fix #43 false positives in header include order by checking includes against a list of c headers.
  Since this interprets certain include lines different than before, output about header include order changes.

1.4.6 (2020-05-31)
-----

* Fix #135: allow 'if constexpr' in readability/braces.
* Fix runtime warning: Close files after reading contents

1.4.5 (2020-01-13)
-----

* Avoid false positive for [build/include_what_you_use] in case of `foo.set<type>` and `foo->set<type>` usage.
* Avoid false positive for [build/include_what_you_use] in case of `map` is user defined function
* Escape backslashes in pydoc strings to get rid of DeprecationWarning.
* Fix false positive "should include its header" for 3rd party headers
* Add support for c++17 tuple destructuring
* fix #123: Inconsistent behavior of --headers and --extensions
* Fix #114: --exclude not working recursively
* fix #112, identifying of copy constructors should allow combinations of volatile and const

1.4.4 (2019-02-25)
-----

Another cleanup release

* NOBUG: fix unit/cli tests for source release
* NOBUG: reduce diff to upstream by intentionally using deprecated functions where upstream uses them
* add `--version` command (https://github.com/cpplint/cpplint/issues/27)

1.4.3 (2019-02-18)
-----

* Revert "Fix the `build/endif_comment` check", same as reverted in upstream

1.4.2 (2019-02-17)
-----

* Cleanup release, fixes further issues with tests and source distribution

1.4.1 (2019-02-17)
-----

* Cleanup release, only adds test support files to source dist

1.4.0 (2019-02-17)
-----

* Incorporate cpplint updates from google (e5d807c6a0d,  2018-05-03)
  * Fix the `build/endif_comment` check (https://github.com/google/styleguide/pull/169)
  * Teach the explicit constructor check about constexpr (#56)
  * Changed vs7 output format (#57)
  * Remove presubmit check for DISALLOW_* macros (#54)
  * add `--quiet` flag as in upstream (https://github.com/google/styleguide/pull/293)
  * support `--root` argument to run in different folder (https://github.com/google/styleguide/pull/291)
  * Fix 16bit Unicode issue (https://github.com/google/styleguide/issues/337)

1.3.0 (2016-07-12)
-----

* Incorporate cpplint updates from google (6d3a7d8a2, 2016-07-14)
* Add --headers flag to choose which extensions are header files.
* Add regression testsing.

1.2.2 (2016-04-07)
-----

* Fixes bug causing RValue detection with namespaces to fail.

1.2.1 (2016-03-19)
-----

* Fixes error in setup.py.

1.2.0 (2016-03-19)
-----

* Adds `.cu` and `.cuh` as supported file extensions by default.
* Moves the warning "Include the directory when naming .h files" from the `build/include` category to the `build/include_subdir` category.

1.1.0 (2016-02-24)
-----

* Adds quiet option to suppress non error-related output.

1.0.1 (2016-02-12)
-----

* Updates PyPi README.

1.0.0 (2016-02-03)
-----

* Fixes a --repository flag bug.

0.0.9 (2016-01-23)
-----

* Adds the --exclude flag to exclude files from being linted.

0.0.8 (2016-01-18)
-----

* Adds the --repository flag to set the location of the project root for header guard calculations.
* Adds support for ``#pragma once`` as an alternative to header include guards.

0.0.7 (2016-01-07)
-----

* Fixes a Windows include guard bug.
* Adds escaping and more detail to JUnit XML output.

0.0.6 (2015-12-15)
-----

* Adds the --recursive flag.
* Adds JUnit XML output.

0.0.5 (2015-01-04)
-----

* Maintenance release, undoes earlier project folder structure changes to remain as true to upstream as possible.

0.0.4 (2015-01-04)
-----

* Merged with upstream revision r141 (2014-12-04)
* This includes many new checks, see commit messages for details
* This also reverts some renaming of files, to stay close to the original project


0.0.3 (2012-11-24)
-----

* python 3 compatibility

0.0.2 (2012-11-06)
-----

* fixed and extended allowed extensions

0.0.1 (2012-10-13)
-----

* import from googlecode, added setup.py
* imported revision r83 (2012-05-11)
