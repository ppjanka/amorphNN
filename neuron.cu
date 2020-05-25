
#include "main.hu"

using namespace std;

__host__
Neuron::Neuron (string _name) {
    cudaMallocManaged(&name, sizeof(string));
    (*name) = _name;
    // allocate
    dendrites = new Connection* [MAX_DENDRITES];
    cudaMallocManaged(&memblock, (4+MAX_DENDRITES)*sizeof(float*)); //stores everythting
    cudaMallocManaged(&memblock_buffer, 3*sizeof(float));
    state = &(memblock_buffer[0]);
    decay_rate = &(memblock_buffer[1]);
    n_dendrites = &(memblock_buffer[2]);
    // initialize
    (*state) = 0.;
    (*decay_rate) = 10.;
    (*n_dendrites) = 0;
    // connect to memblock
    memblock[0] = state;
    memblock[1] = decay_rate;
    memblock[2] = n_dendrites;
    dendrites_states = &(memblock[3]);
    // synchronize
    cudaDeviceSynchronize();
}
__host__
Neuron::~Neuron () {
    for (int i=0; i<int(*n_dendrites); i++) {
        delete dendrites[i];
    }
    delete dendrites;
    cudaFree(memblock_buffer);
    cudaFree(memblock);
    cudaFree(name);
    // synchronize
    cudaDeviceSynchronize();
}

__device__
void Neuron__time_step (float** memblock) {
    float* state = memblock[0];
    float* decay_rate = memblock[1];
    float* n_dendrites = memblock[2];
    float** dendrites_states = &(memblock[3]);
    // decay previous state
    (*state) *= exp(-1./(*decay_rate));
    // read from the dendrites
    for (int i=0; i<int(*n_dendrites); i++) {
        Neuron__adjust_state(state, *(dendrites_states[i]));
    }
}

__device__
void Neuron__adjust_state (float* state, float dx) {
    (*state) += dx;
}

__device__
float Neuron__get_state (float* state) {
    // use sigmoid activation
    return 1./(1.+exp(-(*state)));
}

__host__
void Neuron::attach_dendrite (Neuron* neuron) {
    cudaDeviceSynchronize();
    dendrites[int(*n_dendrites)] = new Connection(neuron);
    dendrites_states[int(*n_dendrites)] = dendrites[int(*n_dendrites)]->state_queue;
    (*n_dendrites) += 1.0;
    cudaDeviceSynchronize();
}

__host__
ostream& operator<< (ostream& cout, const Neuron& n) {
    cudaDeviceSynchronize();
    cout << "Neuron" << *(n.name) << ": state=" << *(n.state) << ", decay_rate=" << *(n.decay_rate) << ", n_dendrites=" << *(n.n_dendrites) << endl;
    for (int i=0; i<int(*(n.n_dendrites)); i++) {
        cout << *(n.dendrites[i]);
    }
    return cout;
}

// utility kernels
__global__
void adjust_state_single_neuron (float* state, float x) {
    Neuron__adjust_state(state, x);
}
__global__
void time_step_single_neuron (float** neuron_memblock) {
    Neuron__time_step (neuron_memblock);
}