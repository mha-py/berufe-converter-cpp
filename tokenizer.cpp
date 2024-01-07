
#include "tokenizer.h"




std::wstring prep(const std::wstring& s) {
    // Adds an extra token for capital letters while lowering these letters. Replaces `Umlaute` by normal vocal plus Umlaut token.
    // Irgendein Beruf becomes ^irgendein ^beruf
    // Sportärztin becomes ^sport°arztin
    std::wstring t;
    
    std::setlocale(LC_ALL, "");

    for (wchar_t c : s) {
        if (std::iswupper(c)) {
            t += L'^';
            t += std::towlower(c);
        } else {
            t += c;
        }
    }

    size_t pos;
    while ((pos = t.find(L'ä')) != std::wstring::npos) {  // Unicode für 'ä'
        t.replace(pos, 1, L"°a");
    }

    while ((pos = t.find(L'ö')) != std::wstring::npos) {  // Unicode für 'ö'
        t.replace(pos, 1, L"°o");
    }

    while ((pos = t.find(L'ü')) != std::wstring::npos) {  // Unicode für 'ü'
        t.replace(pos, 1, L"°u");
    }

    return t;
}


std::wstring deprep(std::wstring str) {
    // The inverse function of prep


    // Ersetze alle '°a' durch 'ä'
    size_t pos = str.find(L"°a");
    while (pos != std::string::npos) {
        str.replace(pos, 2, L"ä");
        pos = str.find(L"°a", pos + 2);
    }

    // Ersetze alle '°o' durch 'ö'
    pos = str.find(L"°o");
    while (pos != std::string::npos) {
        str.replace(pos, 2, L"ö");
        pos = str.find(L"°o", pos + 2);
    }

    // Ersetze alle '°u' durch 'ü'
    pos = str.find(L"°u");
    while (pos != std::string::npos) {
        str.replace(pos, 2, L"ü");
        pos = str.find(L"°u", pos + 2);
    }

    std::vector<wchar_t> t;
    bool nextupper = false;

    for (wchar_t c : str) {
        if (c == L'^') {
            nextupper = true;
        } else if (nextupper) {
            t.push_back(std::toupper(c));
            nextupper = false;
        } else {
            t.push_back(c);
        }
    }
    return std::wstring(t.begin(), t.end());
}




void tokenize(std::wstring string, Tensor1int& res) {
    string = prep(string);
    res.clear();
    for (char c : string) {
        res.push_back(static_cast<uint8_t>(c));
    }
}

std::wstring detokenize(Tensor1int& tokens) {
    std::wstring string;
    for (int a : tokens) {
        string.push_back(static_cast<wchar_t>(a));
    }
    return deprep(string);
}


void encode(std::wstring s, Tensor1int& res) {
    std::wstring paddedString = L"{" + s + std::wstring(MAX_LEN, L'}');
    tokenize(paddedString, res);
    res.resize(MAX_LEN+1);
}


void argmax2(Tensor2 &x, Tensor1int& res) {
    int P = x.size();
    int N = x[0].size();
    res.resize(P);
    for (int p=0; p<P; p++) {
        float max = x[p][0];
        int nmax = 0;
        for (int n=0; n<N; n++) {
            if (x[p][n] > max) {
                max = x[p][n];
                nmax = n;
            }
        }
        res[p] = nmax;
    }
}


int argmax(Tensor1 &x) {
    int N = x.size();
    float max = x[0];
    int nmax = 0;
    for (int n=0; n<N; n++) {
        if (x[n] > max) {
            max = x[n];
            nmax = n;
        }
    }
    return nmax;
}