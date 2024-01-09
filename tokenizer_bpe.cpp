

#include "tokenizer_bpe.h"

int Tokenizer::LoadBPECodes(std::string fname) {
    
    std::string line;

    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cout << "Couldnt open file" << std::endl;
        return -1;
    }

    while (true) {
        if (!std::getline(file, line)) 
            break;
        pair_a.push_back(line);
        if (!std::getline(file, line)) {
            std::cout << "Error, end of file reached!" << std::endl;
            return -2;
        }
        pair_b.push_back(line);
    }

    file.close();
    return 0;
}

int Tokenizer::LoadDicts(std::string fname) {
    
    std::string line;

    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cout << "Couldnt open file" << std::endl;
        return -1;
    }

    int i = 0;
    while (std::getline(file, line)) {
        i2t.push_back(line);
        t2i[line] = i;
        i++;
    }

    file.close();
    return 0;
}


std::vector<std::string> Split(std::string input, std::string delimiter) {
    // Corresponds to string.split(delim) in python
    std::vector<std::string> result;
    size_t pos = 0;
    while ((pos = input.find(delimiter)) != std::string::npos) {
        std::string token = input.substr(0, pos);
        result.push_back(token);
        input.erase(0, pos + delimiter.length());
    }

    // Füge den Rest des Strings hinzu
    if (!input.empty()) {
        result.push_back(input);
    }
    result.erase(result.begin());
    return result;
}



std::vector<std::string> Tokenizer::BPEncode(std::string _text) {
    // pair_a und pair_b sind die Paare in bpe_codes.
    std::string text;
    text.push_back('`');
    for (auto c : _text) {
        text.push_back(c);
        text.push_back('`');
    }

    for (int i=0; i<pair_a.size(); i++) {
        std::string pair = "`" + pair_a[i] + "`" + pair_b[i] + "`";
        size_t pos;
        while ((pos = text.find(pair)) != std::string::npos) {
            text.replace(pos, pair.size(), "`" + pair_a[i]+pair_b[i] + "`");
        }
    }
    return Split(text, "`");
}




std::string Tokenizer::BPDecode(std::vector<std::string> input) {
    std::string result;
    for (auto s : input)
        result.insert(std::end(result), std::begin(s), std::end(s));
    return result;
}


std::vector<int> Tokenizer::tokenize(std::string text) {
    std::vector<std::string> encoded = BPEncode(text);
    std::vector<int> result;
    for (auto s : encoded)
        result.push_back(t2i[s]);
    return result;
}


std::string Tokenizer::detokenize(std::vector<int> input) {
    std::vector<std::string> encoded;
    for (int i : input)
        encoded.push_back(i2t[i]);
    return BPDecode(encoded);
}