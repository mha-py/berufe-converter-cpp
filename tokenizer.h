#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <iostream>
#include <string>
#include "layers.h"

#define MAX_LEN 80


std::wstring prep(const std::wstring& s);
std::wstring deprep(std::wstring str);

void tokenize(std::wstring s, Tensor1int& res);
std::wstring detokenize(Tensor1int& tokens);

void encode(std::wstring s, Tensor1int& res);

void argmax2(Tensor2 &x, Tensor1int& res);
int argmax(Tensor1 &x);

#endif