
#include "main.hu"

using namespace std;

__host__
Connection::Connection (Neuron* neuron) {
    connected_neuron = neuron;
    // allocate
    int delay_init = 3;
    cudaMallocManaged(&memblock, (3+MAX_QUEUE)*sizeof(float*)); //stores everythting
    cudaMallocManaged(&memblock_buffer, (2+MAX_QUEUE)*sizeof(float));
    multiplier = &(memblock_buffer[0]);
    delay = &(memblock_buffer[1]);
    state_queue = &(memblock_buffer[2]);
    // initialize
    connected_neuron_state = neuron->state;
    (*multiplier) = 1.;
    (*delay) = float(delay_init);
    for (int i=0; i<(*delay); i++)
        state_queue[i] = 0.;
    // connect to memblock
    memblock[0] = connected_neuron_state;
    memblock[1] = multiplier;
    memblock[2] = delay;
    memblock[3] = state_queue;
    // synchronize
    cudaDeviceSynchronize();
}
__host__
Connection::~Connection() {
    cudaFree(memblock_buffer);
    cudaFree(memblock);
    // synchronize
    cudaDeviceSynchronize();
}

__device__
float Connection__time_step (float** connection_memblock) {
    // read from memblock
    float* connected_neuron_state = connection_memblock[0];
    float* multiplier = connection_memblock[1];
    float* delay = connection_memblock[2];
    float* state_queue = connection_memblock[3];
    // calculate output and buffer
    float output = (*multiplier) * state_queue[0];
    // update the state queue
    for (int i=1; i<(*delay); i++) {
        state_queue[i-1] = state_queue[i];
    }
    state_queue[int(*delay)-1] = (*connected_neuron_state);
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

// utility kernels
__global__
void time_step_single_connection (float** connection_memblock) {
    Connection__time_step (connection_memblock);
}