#include "net.h"
#include "neuron.h"
#include <iostream>
#include <vector>

Network::Network(const vector<unsigned> &topology){
    unsigned numberOfLayers = topology.size();
    for(unsigned i = 0; i < numberOfLayers; i++)
    {
        double numberOfOutputs;
        if(i == numberOfLayers - 1){numberOfOutputs = 0;}else{numberOfOutputs=topology[i+1];} // if i eq to numberOfLayers - 1. cuz i starts from 0 while size of topology indexed from 0
        m_layers.push_back(Layer());
        for(unsigned j = 0; j <= topology[i]; j++) // <= cuz we wanna add bias node too.
        {
            m_layers.back().push_back(Neuron(j, numberOfOutputs));
            cout << "mara: a neuron has been created!" << endl;
        }
        m_layers.back().back().setOutputValue(1); // bias
    }
}

void Network::feedForward(const vector<double> &inputValues){

    if(inputValues.size() != m_layers[0].size() - 1){throw "mara: invalid net input";}

    for(unsigned int i = 0; i < inputValues.size(); i++){
        m_layers[0][i].setOutputValue(inputValues[i]);        // Feeding the input layer
    }
    for(unsigned i = 1; i < m_layers.size(); i++) // i starts from 1. cuz we have already fed the input layer.
    {
        Layer &prevLayer = m_layers[i-1];
        for(unsigned j = 0; j < m_layers[i].size() - 1; j++) // - 1 for bias
        {
            m_layers[i][j].feedForward(prevLayer); // calling feedForward func of Neuron's objects.
        }
    }
}

void Network::getNetResult(vector<double> &outputValues){
    outputValues.clear(); // clears the input vector if its not.
    for(unsigned i = 0; i < m_layers.back().size() - 1; i++){ // - 1 for bias
        outputValues.push_back(m_layers.back()[i].getOutputValue());
    }
}
void Network::backProb(vector<double> &targetValues){
    for(unsigned i = 0; i < m_layers.back().size()-1; i++){
        m_layers.back()[i].calcOLayerGradients(targetValues[i]); // calcs gradient of output layer neurons
    }
    for(unsigned i = m_layers.size() - 2; i > 0 ; i--){
        Layer &currentLayer = m_layers[i];
        Layer &nextLayer = m_layers[i + 1];
        for(unsigned j = 0; j < currentLayer.size(); j++){
            currentLayer[j].calcHLayerGradients(nextLayer);
        }
    }
    for(unsigned i = m_layers.size() - 1; i > 0; i--){
        Layer &currentLayer = m_layers[i];
        Layer &prevLayer = m_layers[i-1];
        for(unsigned j = 0; j < currentLayer.size() - 1; j++){
            currentLayer[j].updateWeights(prevLayer); // the whole damn point of neural nets
        }
    }
}
