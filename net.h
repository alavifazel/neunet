#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "neuron.h"
using namespace std; // Yes... i knowww

typedef vector<Neuron> Layer; // Defining Layer as a vector of Neuron (s)

class Network
{
public:
    Network(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputValues);
    void backProb( vector<double> &targetValues);
    void getNetResult(vector<double> &outputValues); // gets the final results of the network
private:
    vector <Layer> m_layers;
};

#endif // NETWORK_H
