/*!
 * This code is developed based on an answer from Omnifarious
 * on the stackoverflow.
 * (https://stackoverflow.com/questions/5288076/base64-encoding-and-decoding-with-openssl)
 */

#pragma once

#include <string>

std::string base64_encode(const ::std::string& bindata);

std::string base64_decode(const ::std::string& ascdata);
