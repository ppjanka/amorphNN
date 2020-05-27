
#include "main.hu"

using namespace std;

int main (void) {

    // initialize random number generator on host and device
    srand(time(NULL));
    curandState *curand_state;
    cudaMalloc(&curand_state, sizeof(curandState));

    Brain bob ("bob", 1000, 100, 8, 4, 0., 10., 1, 10, 0.02);
    adjust_state_single_neuron<<<1,1>>>(bob.neurons[0]->state, 1.0);

    //cout << bob; return 0;

    for (int t=0; t<100; t++) {
        Brain__time_step_connections<<<40,32>>>(bob.n_connections, bob.connection_memblocks);
        cudaDeviceSynchronize(); // wait until all connections updated
        Brain__time_step_neurons<<<40,32>>>(bob.n_neurons, bob.neuron_memblocks, bob.n_inputs);
        cudaDeviceSynchronize(); // wait until all connections updated
        //cout << bob;
        bob.print_output();
        cout << "Time " << t << " finished." << endl << endl;
        Brain__shake_connections<<<40,32>>>(bob.n_connections, bob.connection_memblocks, 0.01, curand_state, time(NULL));
    }

    //cout << bob;

    /*Neuron n1 ("1"), n2 ("2");

    cout << n1 << n2;
    n2.attach_dendrite(&n1);
    Connection* c1 = n2.dendrites[0];
    adjust_state_single_neuron<<<1,1>>>(n1.state, 1.0);
    cout << n1 << n2;

    for (int time=0; time<10; time++) {
        time_step_single_connection<<<1,1>>>(c1->memblock);
        time_step_single_neuron<<<1,1>>>(n1.memblock);
        time_step_single_neuron<<<1,1>>>(n2.memblock);
        cout << n1 << n2;
    }*/

    // cleanup
    //curandDestroyGenerator();

    return 0;
}