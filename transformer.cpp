
#include "transformer.h"

#include <iostream>

Transformer::Transformer() {
    emb = new EmbeddingLayer("emb");
    
    ln1 = new LayerNorm("ln1");
    enc1 = new EncoderBlock("enc1");
    enc2 = new EncoderBlock("enc2");
    enc3 = new EncoderBlock("enc3");
    ln2 = new LayerNorm("ln2");

    ln3 = new LayerNorm("ln3");
    dec1 = new DecoderBlock("dec1");
    dec2 = new DecoderBlock("dec2");
    dec3 = new DecoderBlock("dec3");
    ln4 = new LayerNorm("ln4");
    dense1 = new Linear("dense1");
}


void Transformer::encode(const Tensor1int &x, Tensor2 &x_enc) {
    Tensor2 x1, x2;
    emb->forward(x, x1);
    ln1->forward(x1, x2);
    enc1->forward(x2, x1);
    enc2->forward(x1, x2);
    enc3->forward(x2, x1);
    ln2->forward(x1, x_enc);
}



void Transformer::decode(const Tensor1int &y, const Tensor2 &x_enc, Tensor2 mask, Tensor2 &res) {
    Tensor2 y1, y2;
    emb->forward(y, y1);
    ln3->forward(y1, y2);
    dec1->forward(y2, x_enc, mask, y1);
    dec2->forward(y1, x_enc, mask, y2);
    dec3->forward(y2, x_enc, mask, y1);
    ln4->forward(y1, y2);

    dense1->forward(y2, res);
    Softmax(res);
}



void Transformer::load(std::string fname) {
    std::ifstream file(fname, std::ios::binary);

    if (file.is_open()) {
        emb->load_dat(file);

        ln1->load_dat(file);
        enc1->load_dat(file);
        enc2->load_dat(file);
        enc3->load_dat(file);
        ln2->load_dat(file);

        ln3->load_dat(file);
        dec1->load_dat(file);
        dec2->load_dat(file);
        dec3->load_dat(file);
        ln4->load_dat(file);

        dense1->load_dat(file);

        file.close();
    } else {
        std::cerr << "Die Datei " << fname << " konnte nicht geöffnet werden." << std::endl;
    }
}



void CreateMask(Tensor2 &mask, int P) {
    Resize2(mask, P, P);
    for (int i=0; i<P; i++)
        for (int j=0; j<P; j++)
            if (i>=j)
                mask[i][j] = 1.;
}