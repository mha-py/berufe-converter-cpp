#ifndef PREDICTION_H
#define PREDICTION_H


void predict_mult(std::wstring str, int N, Transformer &net, std::vector<std::wstring> &res, std::vector<float> &ps);
int randomchoice(Tensor1 prob);
void sortCorresponding(std::vector<float> &ps, std::vector<std::wstring> &out);


#endif