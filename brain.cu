
#include "main.hu"

using namespace std;

__host__
Brain::Brain (string _name, int _n_neurons, int avg_connections_per_neuron) {
    // parse parameters
    cudaMallocManaged(&name, sizeof(string));
    (*name) = _name;
    cudaMallocManaged(&n_neurons, sizeof(int));
    (*n_neurons) = _n_neurons;
    cudaMallocManaged(&n_connections, sizeof(int));
    (*n_connections) = _n_neurons * avg_connections_per_neuron;
    // allocate
    neurons = new Neuron* [*n_neurons];
    connections = new Connection* [*n_connections];
    cudaMallocManaged(&neuron_memblocks, (*n_neurons) * sizeof(float**));
    cudaMallocManaged(&connection_memblocks, (*n_connections) * sizeof(float**));
    // create neurons
    for (int i=0; i < (*n_neurons); i++) {
        neurons[i] = new Neuron (to_string(i));
        neuron_memblocks[i] = neurons[i]->memblock;
    }
    // wire the brain
    int j = 0;
    int ci = 0;
    for (int i=0; i < (*n_neurons); i++) {
        for (int k=0; k<avg_connections_per_neuron; k++) {// const conn per neur for now
            // randomly choose the connected neuron
            do {j = rand() % (*n_neurons);} while (j == i);
            neurons[i]->attach_dendrite(neurons[j]);
            connections[ci] = neurons[i]->dendrites[int(*(neurons[i]->n_dendrites))-1];
            connection_memblocks[ci] = connections[ci]->memblock;
            ci++;
        }
    }
    cudaDeviceSynchronize();
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
    cudaFree(n_connections);
    cudaFree(n_neurons);
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
void Brain__time_step_neurons (int* n_neurons, float*** neuron_memblocks) {
    // CUDA setup
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // update neurons
    for (int i=index; i<(*n_neurons); i+=stride) {
        Neuron__time_step(neuron_memblocks[i]);
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