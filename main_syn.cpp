

#include "transformer.h"
#include "layers.h"
#include "tokenizer_bpe.h"
#include "prediction_bpe.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <locale>
// #include <set>

int main() {


    system("chcp 1252"); // Typeset der Konsole ändern damit Umlaute richtig verarbeitet werden
    // std::locale::global(std::locale("de_DE.utf8")); // Typeset Linux

    // Test der prediction
    Transformer net;
    net.load("w_syn_060124r.dat");

    Tokenizer tokenizer;
    tokenizer.LoadBPECodes("bpe_codes.txt");
    tokenizer.LoadDicts("tokens.txt");


    std::wstring out;
    while (true) {
        float p;
        std::string input;
        std::getline(std::cin, input);
        std::vector<std::string> out;
        std::vector<float> ps;
        predict_mult(input, 10, net, tokenizer, out, ps);
        for (int i=0; i<out.size(); i++) {
            std::cout << out[i] << std::endl; //<< std::endl;
            std::cout << ps[i] << std::endl << std::endl;
        }
    }


/*
    // Test von prep
    std::wofstream file("test.txt");
    std::wstring input = L"Ärztin ÄÖÜ";
    std::wstring out = prep(input);
    std::wcout << out << std::endl;
    file << out;

    // test von tokenize
    input = L"{Arzt}";
    out = prep(input);
    Tensor1int tokens;
    tokenize(out, tokens);
    std::cout << tokens[0] << std::endl;
    std::cout << tokens[1] << std::endl;
    std::cout << tokens[2] << std::endl;
    out = detokenize(tokens);
    std::wcout << out << std::endl;
    return 0;*/

/*
    // Test 3 des Netzwerkes
    Transformer net;
    net.load("example2.dat");
    std::wstring xstring = L"Arzt";
    std::wstring ystring = L"Ärztin";
    Tensor1int x, y;
    encode(xstring, x);
    encode(ystring, y);

    Tensor2 mask, z;
    CreateMask(mask, y.size());

    net.forward(x, y, mask, z);
    argmax2(z, x);
    std::wcout << detokenize(x) << std::endl;
    
    return 0;*/


/*
    // Test 2 des Netzwerkes
    Transformer net;
    net.load("example2.dat");
    std::wstring xstring = L"asdf";
    std::wstring ystring = L"jkl";
    Tensor1int x, y;
    encode(xstring, x);
    encode(ystring, y);
    Tensor2 mask, z;
    CreateMask(mask, y.size());

    net.forward(x, y, mask, z);
    argmax2(z, x);
    std::wcout << detokenize(x) << std::endl;
    
    return 0;*/

/*
    // Test 1 des Netzwerkes
    Transformer net;
    net.load("example2.dat");

    Tensor1int x { 123,  94,  97, 114, 122, 116, 125 };
    Tensor1int y { 123,  94,  116, 120 };
    Tensor2 mask, res;
    Resize2(mask, 4, 4);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            if (i>=j)
                mask[i][j] = 1.;
            
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    net.forward(x, y, mask, res);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    std::cout << res[0][0] << std::endl;
    std::cout << res[0][1] << std::endl;
    std::cout << res[0][2] << std::endl;
'/

    /*
    // Test der verschiedenen Layer
    Transformer net;
    net.load("example2.dat");

    Tensor2 x, y, res;
    Resize2(x, 4, 256);
    Resize2(y, 4, 256);
    int n=0;
    for (int i=0; i<4; i++) {
        for (int j=0; j<256; j++) {
            x[i][j] = n  * (1.0 / 1000);
            y[i][j] = (n+1024)  * (1.0 / 1000);
            n++;
        }
    }
    Tensor2 mask;
    Resize2(mask, 4, 4);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            if (i>=j)
                mask[i][j] = 1.;

    // DecoderBlock
    net.dec1->forward(x, y, mask, res);
    std::cout << res[0][0] << std::endl;
    std::cout << res[0][1] << std::endl;
    std::cout << res[0][2] << std::endl;
*/

    /*
    // EncoderBlock
    net.enc1->forward(x, res);
    std::cout << res[0][0] << std::endl;
    std::cout << res[0][1] << std::endl;
    std::cout << res[0][2] << std::endl;
    */

    /*
    // Feed Forward Layer
    net.enc1->ff->forward(x, res);
    std::cout << res[0][0] << std::endl;
    std::cout << res[0][1] << std::endl;
    std::cout << res[0][2] << std::endl;
    */


    /*
    // Test des ladens des Transformers
    Transformer net;
    net.load("example2.dat");
    //std::cout << net.dense1->weight.size() << std::endl;
    //std::cout << net.dense1->weight[0].size() << std::endl;
    //return 0;
    std::cout << net.dense1->weight[0][0] << std::endl;
    */



    /*
    // Test von Att
    Tensor3 q, k, v, o, beta;
    Resize3(q, 2, 4, 4);
    Resize3(k, 2, 4, 4);
    Resize3(v, 2, 4, 4);
    int n=0;
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            for (int l=0; l<4; l++) {
                q[i][j][l] = n * 0.01;
                k[i][j][l] = n * 0.01;
                v[i][j][l] = n * 0.01;
                n += 1;
            }
        }
    }
    Tensor2 mask;
    Resize2(mask, 4, 4);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            if (i>=j)
                mask[i][j] = 1.;

    Att(q, k, v, mask, true, o, beta);
    std::cout << beta[0][1][0] << std::endl;
    std::cout << beta[0][1][1] << std::endl;
    std::cout << beta[0][1][2] << std::endl;
    std::cout << beta[0][1][3] << std::endl;*/


    /*
    // Test des ladens von embedding layer und layer norm layer
    std::ifstream file("example2.dat", std::ios::binary);
    if (file.is_open()) {
        EmbeddingLayer emb("");
        LayerNorm ln("");
        emb.load_dat(file);
        ln.load_dat(file);
        std::cout << emb.weight.size() << std::endl;
        std::cout << emb.weight[0].size() << std::endl;
        std::cout << emb.weight[0][0] << std::endl;
        std::cout << ln.gamma[0] << std::endl;
        std::cout << ln.beta[0] << std::endl;
    }*/





    return 0;
}
