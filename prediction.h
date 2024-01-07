#ifndef PREDICTION_H
#define PREDICTION_H


std::wstring predict(std::wstring str, Transformer &net, bool probablistic, float& p);
int randomchoice(Tensor1 prob);



#endif