#ifndef TRANSFORMER_H
#define TRANSFORMER_H


#include "layers.h"

void CreateMask(Tensor2 &mask, int P);

class Transformer {
public:
    EmbeddingLayer* emb;

    LayerNorm* ln1;
    EncoderBlock* enc1;
    EncoderBlock* enc2;
    EncoderBlock* enc3;
    LayerNorm* ln2;

    LayerNorm* ln3;
    DecoderBlock* dec1;
    DecoderBlock* dec2;
    DecoderBlock* dec3;
    LayerNorm* ln4;
    Linear *dense1;

    Transformer();
    void encode(const Tensor1int &x, Tensor2 &x_enc);
    void decode(const Tensor1int &y, const Tensor2 &x_enc, Tensor2 mask, Tensor2 &res);
    void load(std::string fname);

};


#endif