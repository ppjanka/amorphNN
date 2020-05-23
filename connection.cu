
#include "main.hu"

using namespace std;

Connection::Connection (Neuron* neuron) {
    connected_neuron = neuron;
    multiplier = new float;
    *multiplier = 1.;
    delay = new int;
    *delay = 3;
    state_queue = new float [*delay];
    for (int i=0; i<(*delay); i++)
        state_queue[i] = 0.;
}
Connection::~Connection() {
    delete[] state_queue;
    delete multiplier, delay;
}

float Connection::time_step () {
    float output = (*multiplier) * state_queue[0];
    // update the state queue
    #pragma omp simd
    for (int i=1; i<(*delay); i++) {
        state_queue[i-1] = state_queue[i];
    }
    state_queue[(*delay)-1] = connected_neuron->get_state();
    // pass the output to the axion
    return output;
}

ostream& operator<< (ostream& cout, const Connection& c) {
    cout << " --Conn.: multiplier=" << *(c.multiplier) << ", delay=" << *(c.delay) << ", queue: ";
    for (int i=0; i<*(c.delay); i++)
        cout << c.state_queue[i] << " ";
    cout << endl;
    return cout;
}