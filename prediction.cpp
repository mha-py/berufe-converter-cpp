
#include "transformer.h"
#include "tokenizer.h"
#include <math.h> // sqrt
#include <random>
#include <ctime>

int randomchoice(Tensor1 prob) {
    // Gibt bei gegebenen Wahrscheinlichkeiten einen Zufälligen int aus
    // entspricht np.random.choice(range(len(prob)), p=prob) in Python
    static std::mt19937 gen(std::time(0));
    std::discrete_distribution<> d(prob.begin(), prob.end());
    int result = d(gen);
    return result;
}


std::wstring predict(std::wstring str, Transformer &net, bool probablistic, float &p) {
    Tensor1int x, y;
    Tensor2 mask, z, x_enc;
    encode(str, x);
    y.push_back(static_cast<uint8_t>(L'{')); // start token
    int stoptoken = static_cast<uint8_t>(L'}');

    net.encode(x, x_enc);
    int k;

    p = 1;
    int length = MAX_LEN;
    for (int i=0; i<length; i++) {
        CreateMask(mask, y.size());
        net.decode(y, x_enc, mask, z);
        if (probablistic)
            k = randomchoice(z[i]);
        else
            k = argmax(z[i]);
        if (k==stoptoken)
            break;
        y.push_back(k);
        p *= z[i][k];
    }
    std::wstring s = detokenize(y);
    s.erase(0, 1); // remove start token
    return s;
}
