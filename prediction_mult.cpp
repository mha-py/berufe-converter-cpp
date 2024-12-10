
#include "transformer.h"
#include "tokenizer.h"
#include "prediction_syn.h"
#include <math.h> // sqrt
#include <random>
#include <ctime>
#include <algorithm>
#include <numeric> // sort

// SPEZIALFALL FÜR SYNONYME / ES WERDEN MEHRERE OUTPUTS AUSGEGEBEN

int randomchoice(Tensor1 prob) {
    // Gibt bei gegebenen Wahrscheinlichkeiten einen Zufälligen int aus
    // entspricht np.random.choice(range(len(prob)), p=prob) in Python
    static std::mt19937 gen(std::time(0));
    std::discrete_distribution<> d(prob.begin(), prob.end());
    int result = d(gen);
    return result;
}


void predict_mult(std::wstring str, int N, Transformer &net, std::vector<std::wstring> &res, std::vector<float> &ps) {
    Tensor1int x, y;
    Tensor2 mask, z, x_enc;
    encode(str, x);
    y.push_back(static_cast<uint8_t>(L'{')); // start token
    int stoptoken = static_cast<uint8_t>(L'}'); // stop token

    net.encode(x, x_enc);
    int k;
    bool probablistic = true;

    res.resize(0);
    while (res.size()<N) {
        y.resize(1);
        float p = 1;
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
        
        if (std::find(res.begin(), res.end(), s) == res.end()) {
            ps.push_back(p);
            res.push_back(s);
        }
    }
    sortCorresponding(ps, res);
}





void sortCorresponding(std::vector<float> &ps, std::vector<std::wstring> &out) {
    // Erstellen eines Index-Vektors
    std::vector<size_t> idx(ps.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sortieren der Indizes basierend auf den Werten in 'ps'
    std::sort(idx.begin(), idx.end(),
              [&ps](size_t i1, size_t i2) { return ps[i1] > ps[i2]; });

    // Anwenden der sortierten Reihenfolge auf beide Vektoren
    std::vector<float> ps_sorted(ps.size());
    std::vector<std::wstring> out_sorted(out.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        ps_sorted[i] = ps[idx[i]];
        out_sorted[i] = out[idx[i]];
    }

    // Aktualisieren der Originalvektoren
    ps = std::move(ps_sorted);
    out = std::move(out_sorted);
}