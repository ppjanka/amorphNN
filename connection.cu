
#include "main.hu"

using namespace std;

__host__
Connection::Connection (Neuron* neuron) {
    connected_neuron = neuron;
    cudaMallocManaged(&connected_neuron_state, sizeof(float));
    connected_neuron_state = neuron->state;
    cudaMallocManaged(&multiplier, sizeof(float));
    (*multiplier) = 1.;
    cudaMallocManaged(&delay, sizeof(int));
    (*delay) = 3;
    cudaMallocManaged(&state_queue, (*delay)*sizeof(float));
    for (int i=0; i<(*delay); i++)
        state_queue[i] = 0.;
    cudaDeviceSynchronize();
}
__host__
Connection::~Connection() {
    cudaFree(state_queue);
    cudaFree(multiplier);
    cudaFree(delay);
}

__device__
float Connection__time_step (float* multiplier, int* delay, float* state_queue, float* connected_neuron_state) {
    float output = (*multiplier) * state_queue[0];
    // update the state queue
    for (int i=1; i<(*delay); i++) {
        state_queue[i-1] = state_queue[i];
    }
    state_queue[(*delay)-1] = (*connected_neuron_state);
    // pass the output to the axion
    return output;
}

__host__
ostream& operator<< (ostream& cout, const Connection& c) {
    cudaDeviceSynchronize();
    cout << " --Conn.: multiplier=" << *(c.multiplier) << ", delay=" << *(c.delay) << ", queue: ";
    for (int i=0; i<*(c.delay); i++)
        cout << c.state_queue[i] << " ";
    cout << endl;
    return cout;
}