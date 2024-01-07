
#include "layers.h"
#include <math.h> // sqrt
#include <iostream>
#include <cmath> // pow

void Resize2(Tensor2 &x, int N, int M) {
    x.resize(N);
    for (int i=0; i<N; i++)
        x[i].resize(M);
}

void Resize3(Tensor3 &x, int N, int M, int L) {
    x.resize(N);
    for (int i=0; i<N; i++)
        Resize2(x[i], M, L);
}


void EmbeddingLayer::forward(const Tensor1int x, Tensor2 &res) {
    int P = x.size();
    int I = weight.size();
    int J = weight[0].size();

    Resize2(res, P, J);

    #pragma omp parallel for // Schleife rechnet parallel
    for (int p=0; p<P; p++) {
        for (int j=0; j<J; j++) {
            res[p][j] = weight[x[p]][j];
        }
    }
}

void EmbeddingLayer::load_dat(std::ifstream &file) {
    int N, M;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    file.read(reinterpret_cast<char*>(&M), sizeof(M));
    Resize2(weight, N, M);
    
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            file.read(reinterpret_cast<char*>(&weight[i][j]), sizeof(float));
        }
    }
}



void LayerNorm::forward(const Tensor2 x, Tensor2 &res) {
    int P = x.size();
    int N = x[0].size();

    Resize2(res, P, N);

    #pragma omp parallel for // Schleife rechnet parallel
    for (int p=0; p<P; p++) {
        float mean = 0;
        float var = 0;
        for (int n=0; n<N; n++) {
            mean += x[p][n];
            var += x[p][n] * x[p][n];
        }
        mean /= N;
        var /= N;
        var = var - mean*mean;
        float std = std::sqrt(var + 1e-12);

        for (int n=0; n<N; n++) {
            res[p][n] = (x[p][n] - mean) / std * gamma[n] + beta[n];
        }
    }
}

void LayerNorm::load_dat(std::ifstream &file) {
    
    // gamma
    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    gamma.resize(N);
    for (int i=0; i<N; i++) {
        file.read(reinterpret_cast<char*>(&gamma[i]), sizeof(float));
    }
    
    // beta
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    beta.resize(N);
    for (int i=0; i<N; i++) {
        file.read(reinterpret_cast<char*>(&beta[i]), sizeof(float));
    }
}



void Linear::forward(const Tensor2 x, Tensor2 &res) {
    int P = x.size();
    int I = x[0].size();
    int J = weight.size();

    Resize2(res, P, J);

    #pragma omp parallel for // Schleife rechnet parallel
    for (int p=0; p<P; p++) {
        for (int j=0; j<J; j++) {
            res[p][j] = bias[j];
            for (int i=0; i<I; i++) {
                res[p][j] += weight[j][i] * x[p][i];
            }
        }
    }
}


void Linear::load_dat(std::ifstream &file) {
    // Weight
    int N, M;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    file.read(reinterpret_cast<char*>(&M), sizeof(M));
    Resize2(weight, N, M);
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            file.read(reinterpret_cast<char*>(&weight[i][j]), sizeof(float));
        }
    }
    
    // Bias
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    bias.resize(N);
    for (int i=0; i<N; i++) {
        file.read(reinterpret_cast<char*>(&bias[i]), sizeof(float));
    }
}




void Att(const Tensor3 &q, const Tensor3 &k, const Tensor3 &v,
                        const Tensor2 &mask, bool bias, Tensor3 &o, Tensor3 &beta) {
    int H = q.size(); 
    int I = q[0].size();
    int M = q[0][0].size();
    int J = k[0].size();
    int N = v[0][0].size();

    // beta
    Resize3(beta, H, I, J);
    float sqrt_M = std::sqrt(1.0*M);
    #pragma omp parallel for // Schleife rechnet parallel
    for (int h=0; h<H; h++) {
        for (int i=0; i<I; i++) {
            for (int j=0; j<J; j++) {
                beta[h][i][j] = 0;
                for (int m=0; m<M; m++) {
                    beta[h][i][j] += q[h][i][m] * k[h][j][m] / sqrt_M;
                }
            }
        }
    }

    // Bias
    if (bias) {
        float x = pow(pow(2, 8), 1.0 / H);
        #pragma omp parallel for // Schleife rechnet parallel
        for (int h=0; h<H; h++) {
            float slope = 1 / pow(x, (h+1));
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++) {
                    beta[h][i][j] -= slope * std::abs(i-j);
                }
            }
        }
    }
    
    // Mask
    if (mask.size() > 0) {
        #pragma omp parallel for // Schleife rechnet parallel
        for (int h=0; h<H; h++) {
            for (int i=0; i<I; i++) {
                for (int j=0; j<J; j++) {
                    if (mask[i][j] == 0.) {
                        beta[h][i][j] = -99999.;
                    }
                }
            }
        }
    }


    // softmax
    #pragma omp parallel for // Schleife rechnet parallel
    for (int h=0; h<H; h++) {
        for (int i=0; i<I; i++) {
            float sum = 0;
            for (int j=0; j<J; j++) {
                float a = std::exp(beta[h][i][j]);
                beta[h][i][j] = a;
                sum += a;
            }
            for (int j=0; j<J; j++) {
                beta[h][i][j] /= sum;
            }
        }
    }

    // output
    Resize3(o, H, I, N);
    #pragma omp parallel for // Schleife rechnet parallel
    for (int h=0; h<H; h++) {
        for (int i=0; i<I; i++) {
            for (int n=0; n<N; n++) {
                o[h][i][n] = 0;
                for (int j=0; j<J; j++) {
                    o[h][i][n] += beta[h][i][j] * v[h][j][n];
                }
            }
        }
    }
}




void Rearrange(Tensor2 &x, int H, Tensor3 &res) {
    int P = x.size();
    int N = x[0].size();
    N = N / H;
    Resize3(res, H, P, N);
    #pragma omp parallel for // Schleife rechnet parallel
    for (int h=0; h<H; h++)
        for (int p=0; p<P; p++)
            for (int n=0; n<N; n++)
                res[h][p][n] = x[p][N*h+n];
}

void Rearrange2(Tensor3 &x, Tensor2 &res) {
    int H = x.size();
    int P = x[0].size();
    int N = x[0][0].size();
    Resize2(res, P, H*N);
    #pragma omp parallel for // Schleife rechnet parallel
    for (int h=0; h<H; h++)
        for (int p=0; p<P; p++)
            for (int n=0; n<N; n++)
                res[p][N*h+n] = x[h][p][n];
}



MultiHeadAttention::MultiHeadAttention(std::string _name) {
    name = _name;
    qlayer = new Linear(name + "/q");
    klayer = new Linear(name + "/k");
    vlayer = new Linear(name + "/v");
    player = new Linear(name + "/p");
}



void MultiHeadAttention::forward(const Tensor2 x, const Tensor2 y, const Tensor2 z, const Tensor2 mask, Tensor2 &res) {

    Tensor2 q, k, v;
    qlayer->forward(x, q);
    klayer->forward(y, k);
    vlayer->forward(z, v);

    Tensor3 q2, k2, v2;
    Rearrange(q, nh, q2);
    Rearrange(k, nh, k2);
    Rearrange(v, nh, v2);

    Tensor3 o, beta;
    Att(q2, k2, v2, mask, true, o, beta);

    Tensor2 o2;
    Rearrange2(o, o2);

    player->forward(o2, res);
}


void MultiHeadAttention::load_dat(std::ifstream &file) {
    qlayer->load_dat(file);
    klayer->load_dat(file);
    vlayer->load_dat(file);
    player->load_dat(file);
    file.read(reinterpret_cast<char*>(&nh), sizeof(nh));
}


float act(float x) {
    // Activation function
    return std::max(x, float(0.)); // RELU
    //return 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

void Activation(Tensor2 &x) {
    // Applies 'act' on every element of x
    int P = x.size();
    int N = x[0].size();
    for (int p=0; p<P; p++)
        for (int n=0; n<N; n++)
            x[p][n] = act(x[p][n]);
}

void Add(Tensor2& x, const Tensor2& y) {
    // Inplace Addition
    int P = x.size();
    int N = x[0].size();
    for (int p=0; p<P; p++)
        for (int n=0; n<N; n++)
            x[p][n] += y[p][n];
}




FeedForwardLayer::FeedForwardLayer(std::string _name) {
    name = _name;
    ln = new LayerNorm(name + "/ln");
    dense1 = new Linear(name + "/dense1");
    dense2 = new Linear(name + "/dense2");
}



void FeedForwardLayer::forward(const Tensor2 x, Tensor2 &res) {
    Tensor2 y1, y2;
    ln->forward(x, y1);
    dense1->forward(y1, y2);
    Activation(y2);
    dense2->forward(y2, res);
    Add(res, x);
}


void FeedForwardLayer::load_dat(std::ifstream &file) {
    ln->load_dat(file);
    dense1->load_dat(file);
    dense2->load_dat(file);
}




EncoderBlock::EncoderBlock(std::string name) {
    mha = new MultiHeadAttention(name + "/mha");
    ln = new LayerNorm(name + "/ln");
    ff = new FeedForwardLayer(name + "/ff");
}



void EncoderBlock::forward(const Tensor2 x, Tensor2 &res) {
    Tensor2 y1, y2, nomask;
    ln->forward(x, y1);
    mha->forward(y1, y1, y1, nomask, y2);
    Add(y2, x);
    ff->forward(y2, res);
}


void EncoderBlock::load_dat(std::ifstream &file) {
    mha->load_dat(file);
    ln->load_dat(file);
    ff->load_dat(file);
}




DecoderBlock::DecoderBlock(std::string name) {
    mha1 = new MultiHeadAttention(name + "/mha1");
    mha2 = new MultiHeadAttention(name + "/mha2");
    ln1 = new LayerNorm(name + "/ln1");
    ln2 = new LayerNorm(name + "/ln2");
    ff = new FeedForwardLayer(name + "/ff");
}



void DecoderBlock::forward(const Tensor2 x, const Tensor2 y, const Tensor2 mask, Tensor2 &res) {
    Tensor2 z, x1, x2;
    ln1->forward(x, z);
    mha1->forward(z, z, z, mask, x1); // self attention
    Add(x1, x);
    ln2->forward(x1, z);
    Tensor2 nomask;
    mha2->forward(z, y, y, nomask, x2); // cross attention
    Add(x2, x1);
    ff->forward(x2, res);
}



void DecoderBlock::load_dat(std::ifstream &file) {
    mha1->load_dat(file);
    mha2->load_dat(file);
    ln1->load_dat(file);
    ln2->load_dat(file);
    ff->load_dat(file);
}


void Softmax(Tensor2 &x) {
    int P = x.size();
    int N = x[0].size();
    for (int p=0; p<P; p++) {
        float sum = 0;
        for (int n=0; n<N; n++) {
            float a = std::exp(x[p][n]);
            x[p][n] = a;
            sum += a;
        }
        for (int n=0; n<N; n++) {
            x[p][n] /= sum;
        }
    }
}