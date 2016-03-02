# Neural Net
Repo contains a Class for building a neural network.
example for the classcial XOR operator:
  
  ``` c++
  
#include <iostream>
#include <vector>
#include "net.h"
#include <ctime>
#include <iostream>
#include <cstdlib>

using namespace std;
int randomValue();
void showVectorVals(string label, vector<double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main()
{
    srand ( time(NULL) );
    vector <unsigned> topology = {2,4,1}; // For classical XOR operator
    Network myNet(topology);
    for(unsigned i = 0; i < 4000; i++){
        vector <double> input, output, result;
	// messy but works for a demo
        int n1,n2,o;
        n1 = randomValue();
        n2 = randomValue();
        o = n1 ^ n2;
        input.push_back(n1);
        input.push_back(n2);
        output.push_back(o);
        showVectorVals("input: ", input);
        myNet.feedForward(input);

        myNet.getNetResult(result);
        showVectorVals("output: ", result);

        myNet.backProb(output);
        showVectorVals("out: ", output);
    }

    return 0;
}

int randomValue(){
    return (int)(2.0 * rand() / double(RAND_MAX));
}


  ```
  
  Last lines of output on my machine:
  ```
  .
  .
  .
output:  -0.0016593 
out:  0 
input:  1 1 
output:  0.00231707 
out:  0 
input:  0 1 
output:  0.985709 
out:  1 
input:  1 1 
output:  0.000977401 
out:  0 
input:  1 0 
output:  0.984539 
out:  1
