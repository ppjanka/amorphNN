
#include "main.hu"

using namespace std;

__host__
Neuron::Neuron (string _name) {
    cudaMallocManaged(&name, sizeof(string));
    (*name) = _name;
    cudaMallocManaged(&state, sizeof(float));
    (*state) = 0.;
    cudaMallocManaged(&decay_rate, sizeof(float));
    (*decay_rate) = 10.;
    cudaMallocManaged(&n_dendrites, sizeof(int));
    (*n_dendrites) = 0;
    cudaMallocManaged(&dendrites, MAX_DENDRITES * sizeof(Connection*));
    cudaMallocManaged(&dendrites_states, MAX_DENDRITES * sizeof(float));
    cudaDeviceSynchronize();
}
__host__
Neuron::~Neuron () {
    for (int i=0; i<*n_dendrites; i++) {
        delete dendrites[i];
    }
    cudaFree(dendrites);
    cudaFree(dendrites_states);
    cudaFree(name);
    cudaFree(state);
    cudaFree(decay_rate);
    cudaFree(n_dendrites);
}

__device__
void Neuron__time_step (float* state, float* decay_rate, int* n_dendrites, float** dendrites_states) {
    // decay previous state
    (*state) *= exp(-1./(*decay_rate));
    // read from the dendrites
    for (int i=0; i<(*n_dendrites); i++) {
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
    dendrites[*n_dendrites] = new Connection(neuron);
    dendrites_states[*n_dendrites] = dendrites[*n_dendrites]->state_queue;
    (*n_dendrites)++;
    cudaDeviceSynchronize();
}

__host__
ostream& operator<< (ostream& cout, const Neuron& n) {
    cudaDeviceSynchronize();
    cout << "Neuron" << *(n.name) << ": state=" << *(n.state) << ", decay_rate=" << *(n.decay_rate) << ", n_dendrites=" << *(n.n_dendrites) << endl;
    for (int i=0; i<*(n.n_dendrites); i++) {
        cout << *(n.dendrites[i]);
    }
    return cout;
}