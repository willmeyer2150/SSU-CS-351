
#include <barrier>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// Header file for the Data template class
#include "Data.h"

/////////////////////////////////////////////////////////////////////////////
//
// --- main ---
//
int main(int argc, char* argv[]) {
    //
    // Specify the program's default options:
    //   * the default filename
    //   * the default number of threads
    //
    std::string filename = "data.bin";
    size_t numThreads = 4;

    //-----------------------------------------------------------------------
    //
    // Process command-line options.  Accepted options are specified in the
    //   character string "options".  All options are single characters
    //   (as compared to full words like some programs).  If a character
    //   is followed by a color (:), the option expects an additional
    //   parameter.
    //
    // Options are processed using a simple library function getopt()
    //
    int option;
    const char* options = "f:ht:";
    while ((option = getopt(argc, argv, options)) != -1) {
        switch (option) {
            case 'h': {
                const char* help =
                    "Usage: %s -[fht]\n"
                    "    -h           show help message\n"
                    "    -f <name>    read data from <name>\n"
                    "    -t <value>   use <values> number of threads (default: %u)\n";

                    fprintf(stderr, help, argv[0], numThreads);
                    exit(EXIT_SUCCESS);
            } break;

            case 'f':
                filename = optarg;
                break;

            case 't':
                numThreads = std::stol(optarg);
                break;
        }
    }

    //-----------------------------------------------------------------------
    //
    // Access our the data file through our Data C++ class.  Under the hood,
    //   this class uses an advanced file-access technique called memory
    //   mapping, which makes the file looked like an array (although our
    //   Data class makes it look more like a std::vector), allowing indexed
    //   random-access to the data.
    //
    Data<float>  data(filename.c_str());

    //-----------------------------------------------------------------------
    //
    // A collection of variables to make threading the application simpler.
    //
    //   * threads - is merely an array of std::jthreads to store the created
    //       threads, similar to what was demonstrated in class
    //   * sums - provides a per-thread sum allowing you to accumulate the
    //       values computed in a thread independent of other threads
    //   * barrier - provides a synchronization barrier to prevent threads
    //       exiting before their peers
    //
    std::vector<std::jthread>  threads(numThreads);
    std::vector<double>        sums(numThreads);
    std::barrier               barrier(numThreads);

    // Computation of how much work a thread should do.
    size_t chunkSize = (data.size() / numThreads) + 1;

    //-----------------------------------------------------------------------
    //
    // The computational kernel, where you should enter your thread
    //   implementation.  Much of the thread framework is provided below,
    //   and you mostly needed to compute array bounds to be processed in
    //   the threads, the computational loop (see mean.cpp), and lambda's
    //   closure configuration.
    //
    for (size_t id = 0; id < threads.size(); ++id) {
        threads[id] = std::jthread(
            []() {
                // Add your implementation here

                barrier.arrive_and_wait();
            }
        );
    }

    //-----------------------------------------------------------------------
    //
    // The main thread's final work.  As discussed in class, we wait on
    //   the last thread explicitly (because we're using std::jthreads
    //   recall that we've already joined them at their creation), but this
    //   line pauses the main thread until the last thread terminates
    threads.back().join();

    //-----------------------------------------------------------------------
    //
    // Compute the final sum by tallying the values from each thread, and
    //   report the results
    //
    double sum = std::accumulate(std::begin(sums), std::end(sums), 0.0);

    std::cout << "Samples = " << data.size() << "\n";
    std::cout << "Mean = " << sum / data.size() << "\n";
}
