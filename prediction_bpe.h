#ifndef PREDICTION_H
#define PREDICTION_H


void predict_mult(std::string str, int N, Transformer &net, Tokenizer &tokenizer, std::vector<std::string> &res, std::vector<float> &ps);
int randomchoice(Tensor1 prob);
void sortCorresponding(std::vector<float> &ps, std::vector<std::wstring> &out);


#endif