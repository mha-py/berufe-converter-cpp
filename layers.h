#ifndef LAYERS_H
#define LAYERS_H


#include <string>
#include <vector>
#include <fstream>

typedef std::vector<uint8_t> Tensor1int;
typedef std::vector<float> Tensor1;
typedef std::vector<Tensor1> Tensor2;
typedef std::vector<Tensor2> Tensor3;

void Resize2(Tensor2 &x, int N, int M);
void Resize3(Tensor3 &x, int N, int M, int L);


void Att(const Tensor3 &q, const Tensor3 &k, const Tensor3 &v,
                        const Tensor2 &mask, bool bias, Tensor3 &o, Tensor3 &beta);

void Rearrange(Tensor2 &x, int H, Tensor3 &res);
void Rearrange2(Tensor3 &x, Tensor2 &res);

void Activation(Tensor2 &x);

void Softmax(Tensor2& x);


class EmbeddingLayer {
public:
    std::string name;
    Tensor2 weight;

    EmbeddingLayer(std::string name) : name(name) {};
    void forward(const Tensor1int x, Tensor2 &res);
    void load_dat(std::ifstream &file);
};


class LayerNorm {
public:

    std::string name;
    Tensor1 gamma, beta;

    LayerNorm(std::string name) : name(name) {};
    void forward(const Tensor2 x, Tensor2 &res);
    void load_dat(std::ifstream &file);
};



class Linear {
public:

    std::string name;
    Tensor2 weight;
    Tensor1 bias;

    Linear(std::string name) : name(name) {};
    void forward(const Tensor2 x, Tensor2 &res);
    void load_dat(std::ifstream &file);
};




class MultiHeadAttention {
public:

    std::string name;
    Linear* qlayer;
    Linear* klayer;
    Linear* vlayer;
    Linear* player;
    int nh;

    MultiHeadAttention(std::string _name);
    void forward(const Tensor2 x, const Tensor2 y, const Tensor2 z, const Tensor2 mask, Tensor2 &res);
    void load_dat(std::ifstream &file);
};




class FeedForwardLayer {
public:

    std::string name;
    LayerNorm* ln;
    Linear* dense1;
    Linear* dense2;

    FeedForwardLayer(std::string name);
    void forward(const Tensor2 x, Tensor2 &res);
    void load_dat(std::ifstream &file);
};





class EncoderBlock {
public:

    MultiHeadAttention* mha;
    LayerNorm* ln;
    FeedForwardLayer* ff;

    EncoderBlock(std::string name);
    void forward(const Tensor2 x, Tensor2 &res);
    void load_dat(std::ifstream &file);
};






class DecoderBlock {
public:

    MultiHeadAttention* mha1;
    MultiHeadAttention* mha2;
    LayerNorm* ln1;
    LayerNorm* ln2;
    FeedForwardLayer* ff;

    DecoderBlock(std::string name);
    void forward(const Tensor2 x, const Tensor2 y, const Tensor2 mask, Tensor2 &res);
    void load_dat(std::ifstream &file);
};







#endif