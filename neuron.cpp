#include "neuron.h"
#include <iostream>
Neuron::Neuron(unsigned index, unsigned numberOfOutputs){
    m_index = index;
    for(unsigned i = 0; i < numberOfOutputs; i++){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = rand() / double(RAND_MAX);
    }
}

void Neuron::feedForward(Layer &prevLayer){
    double sum = 0.0;
    for(unsigned i = 0; i < prevLayer.size(); i++){
        sum += prevLayer[i].getOutputValue() * prevLayer[i].m_outputWeights[m_index].weight;   
    }
    m_outputValue = activationFunc(sum);
}

void Neuron::calcOLayerGradients(double targetValue){
    double delta = targetValue - m_outputValue; // fix it
    m_gradient = delta * Neuron::activationFuncDerivative(m_outputValue);
}

void Neuron::calcHLayerGradients(Layer &nextLayer){
    double sum = 0.0;
    for(unsigned i = 0; i < nextLayer.size() - 1; i++){ // - 1 for bias
        sum += nextLayer[i].m_gradient * m_outputWeights[i].weight;
    }
    m_gradient = sum * Neuron::activationFuncDerivative(m_outputValue);
}

void Neuron::updateWeights(Layer &prevLayer){
    for(unsigned i = 0; i < prevLayer.size(); i++){
        double newDeltaWeight = eta * prevLayer[i].getOutputValue() * m_gradient + alpha * prevLayer[i].m_outputWeights[m_index].deltaWeight;
        prevLayer[i].m_outputWeights[m_index].deltaWeight = newDeltaWeight;
        prevLayer[i].m_outputWeights[m_index].weight += newDeltaWeight;
    }
}
