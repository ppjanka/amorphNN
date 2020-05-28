
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

    bob.flicker_test(40,32,10000,curand_state,0.001);

    //return 0;

    // setup input
    for (int i=0; i<8; i++) {
        (*(bob.input_states[i])) = rand()/RAND_MAX;
    }
    // setup label to memorize
    float* label = new float [4];
    for (int i=0; i<4; i++) {
        label[i] = 0;
    }
    label[0] = 1;

    //bob.train(label,40,32,1e6,curand_state,0.01);

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

    //cleanup
    delete label;

    return 0;
}