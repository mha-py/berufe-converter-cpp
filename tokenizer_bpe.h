#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <regex>


class Tokenizer {
public:
    std::vector<std::string> pair_a, pair_b; // pairs of the bpe encoding
    std::vector<std::string> i2t; // int to token (string)
    std::map<std::string, int> t2i; // token (string) to int
    LoadBPECodes(std::string fname); // loads i2t and t2i
    LoadDicts(std::string fname);

    std::vector<std::string> BPEncode(std::string _text);
    std::string BPDecode(std::vector<std::string> input);

    std::vector<int> tokenize(std::string text);
    std::string detokenize(std::vector<int> input);
};



#endif