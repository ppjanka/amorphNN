
#include "main.hu"

using namespace std;

__host__
Brain::Brain (string _name, int _n_neurons, int avg_connections_per_neuron, int _n_inputs, int _n_outputs, float max_init_state, float max_init_delay) {
    cout << "Creating " << _name << "'s brain..." << endl;
    // parse parameters
    cout << " -- parsing parameters.. " << flush;
    cudaMallocManaged(&name, sizeof(string));
    (*name) = _name;
    cudaMallocManaged(&n_neurons, sizeof(int));
    (*n_neurons) = _n_neurons;
    cudaMallocManaged(&n_connections, sizeof(int));
    (*n_connections) = (_n_neurons-_n_inputs) * avg_connections_per_neuron;
    cudaMallocManaged(&n_inputs, sizeof(int));
    (*n_inputs) = _n_inputs;
    cudaMallocManaged(&n_outputs, sizeof(int));
    (*n_outputs) = _n_outputs;
    cout << "done." << endl;
    // allocate
    cout << " -- allocating memory.. " << flush;
    cudaMallocManaged(&input_states, (*n_inputs) * sizeof(float*));
    cudaMallocManaged(&output_states, (*n_outputs) * sizeof(float*));
    neurons = new Neuron* [*n_neurons];
    connections = new Connection* [*n_connections];
    cudaMallocManaged(&neuron_memblocks, (*n_neurons) * sizeof(float**));
    cudaMallocManaged(&connection_memblocks, (*n_connections) * sizeof(float**));
    cout << "done." << endl;
    // create neurons
    cout << " -- creating neurons.. " << flush;
    for (int i=0; i < (*n_neurons); i++) {
        // use random state and decay rate
        neurons[i] = new Neuron (to_string(i), max_init_state*rand()/RAND_MAX, max_init_delay*rand()/RAND_MAX);
        neuron_memblocks[i] = neurons[i]->memblock;
    }
    input_neurons = neurons;
    output_neurons = &(neurons[(*n_neurons)-(*n_outputs)-1]);
    for (int i=0; i < (*n_inputs); i++) {
        input_states[i] = neurons[i]->state;
    }
    int istart = (*n_neurons-*n_outputs);
    for (int i=istart; i < (*n_neurons); i++) {
        output_states[i-istart] = neurons[i]->state;
    }
    cout << "done." << endl;
    // wire the brain
    cout << " -- wiring the brain.. " << flush;
    int j = 0;
    int ci = 0;
    for (int i=(*n_inputs); i < (*n_neurons); i++) { // don't connect input neurons
        for (int k=0; k<avg_connections_per_neuron; k++) {// const conn per neur for now
            // randomly choose the connected neuron
            do {j = rand() % (*n_neurons);} while (j == i);
            neurons[i]->attach_dendrite(neurons[j]);
            connections[ci] = neurons[i]->dendrites[int(*(neurons[i]->n_dendrites))-1];
            connection_memblocks[ci] = connections[ci]->memblock;
            ci++;
        }
    }
    cout << "done." << endl;
    cout << " -- synchronizing.. " << flush;
    cudaDeviceSynchronize();
    cout << "done." << endl;
    cout << _name << "'s brain is ready!" << endl << endl;
}

__host__
Brain::~Brain () {
    for (int i=0; i<(*n_neurons); i++) {
        delete neurons[i]; // also deletes connections
    }
    delete neurons;
    delete connections;
    cudaFree(connection_memblocks);
    cudaFree(neuron_memblocks);
    cudaFree(input_states);
    cudaFree(output_states);
    cudaFree(n_connections);
    cudaFree(n_neurons);
    cudaFree(n_inputs);
    cudaFree(n_outputs);
    cudaFree(name);
    // synchronize
    cudaDeviceSynchronize();
}

__global__
void Brain__time_step_connections (int* n_connections, float*** connection_memblocks) {
    // CUDA setup
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // update connections
    for (int i=index; i<(*n_connections); i+=stride) {
        Connection__time_step(connection_memblocks[i]);
    }
}
__global__
void Brain__time_step_neurons (int* n_neurons, float*** neuron_memblocks, int* n_inputs) {
    // CUDA setup
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // update neurons (ignore inputs)
    for (int i=index; i<(*n_neurons-*n_inputs); i+=stride) {
        Neuron__time_step(neuron_memblocks[i+(*n_inputs)]);
    }
}

// raise all connection multiplicators to a small random power in (1-eps, 1+eps)
__global__
void Brain__shake_connections (int* n_connections, float*** connection_memblocks, float eps, curandState* curand_state, unsigned long seed) {
    // CUDA setup
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // CUrand setup
    curand_init(seed, index, 0, &curand_state[index]);
    // update connections
    float power;
    for (int i=index; i<(*n_connections); i+=stride) {
        power = (1.-eps) + 2.*eps * curand_normal(curand_state);
        *(connection_memblocks[i][1]) = pow(*(connection_memblocks[i][1]), power);
    }
}
// raise all connection multiplicators to a given constant power
__global__
void Brain__feedback_connections (int* n_connections, float*** connection_memblocks, float power) {
    // CUDA setup
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // update connections
    for (int i=index; i<(*n_connections); i+=stride) {
        *(connection_memblocks[i][1]) = pow(*(connection_memblocks[i][1]), power);
    }
}

__host__
ostream& operator<< (ostream& cout, const Brain& b) {
    cudaDeviceSynchronize();
    cout << "Brain " << *(b.name) << ":" << endl;
    for (int i=0; i<(*(b.n_neurons)); i++) {
        cout << "   " << *(b.neurons[i]);
    }
    cout << endl;
    return cout;
}
__host__
void Brain::print_output () {
    cudaDeviceSynchronize();
    cout << *name << " says: " << flush;
    for (int i=0; i<(*n_outputs); i++) {
        cout << *(output_states[i]) << "  " << flush;
    }
    cout << endl;
}