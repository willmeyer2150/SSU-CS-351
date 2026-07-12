
#include <iostream>

// Header file for the Data template class
#include "Data.h"

int main(int argc, char* argv[]) {
    const char* filename = argc < 2 ? "data.bin" : argv[1];

    //-----------------------------------------------------------------------
    //
    // Access our the data file through our Data C++ class.  Under the hood,
    //   this class uses an advanced file-access technique called memory
    //   mapping, which makes the file looked like an array (although our
    //   Data class makes it look more like a std::vector), allowing indexed
    //   random-access to the data.
    //
    Data<float>  data(filename);

    //-----------------------------------------------------------------------
    //
    // The computational kernel that computes the mean by summing the
    //   values in the data array. 
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }

    //-----------------------------------------------------------------------
    //
    // Report the results.
    //
    std::cout << "Samples = " << data.size() << "\n";
    std::cout << "Mean = " << sum / data.size() << "\n";
}

