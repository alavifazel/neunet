#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <cmath>
#include <cstdlib>
using namespace std; // yup i know

class Neuron; // to make the compiler happy.
typedef vector<Neuron> Layer; // Defining Layer as a vector of Neuron(s)

struct Connection{
    double weight, deltaWeight;
};

class Neuron
{
public:
    Neuron(unsigned index, unsigned numberOfOutputs);
    static double activationFunc(double x){return tanh(x);}
    static double activationFuncDerivative(double x){return 1.0 - x * x;}
    void setOutputValue(double x){m_outputValue = x;} // setter func for m_outputValue
    double getOutputValue(){return m_outputValue;}
    void feedForward(Layer &prevLayer);
    void calcOLayerGradients(double targetValue);
    void calcHLayerGradients(Layer &nextLayer);
    void updateWeights(Layer &prevLayer);

private:
    vector<Connection>m_outputWeights;
    unsigned m_index; // it's index in Layer
    double m_outputValue; // output value of neuron
    double m_gradient;
    double eta = 0.25, alpha = 0.5;
};

#endif // NEURON_H
