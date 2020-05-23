#include <iostream>
#include "main.hu"

using namespace std;

__global__
void adjust_state_single_neuron (float* state, float x) {
    Neuron__adjust_state(state, x);
}

__global__
void time_step_single_neuron (float* state, float* decay_rate, int* n_dendrites, float** dendrites_states) {
    Neuron__time_step (state, decay_rate, n_dendrites, dendrites_states);
}

__global__
void time_step_single_connection (float* multiplier, int* delay, float* state_queue, float* connected_neuron_state) {
    Connection__time_step (multiplier, delay, state_queue, connected_neuron_state);
}

int main (void) {

    Neuron n1 ("1"), n2 ("2");

    cout << n1 << n2;
    n2.attach_dendrite(&n1);
    Connection* c1 = n2.dendrites[0];
    adjust_state_single_neuron<<<1,1>>>(n1.state, 1.0);
    cout << n1 << n2;

    for (int time=0; time<10; time++) {
        time_step_single_connection<<<1,1>>>(c1->multiplier, c1->delay, c1->state_queue, c1->connected_neuron_state);
        time_step_single_neuron<<<1,1>>>(n1.state, n1.decay_rate, n1.n_dendrites, n1.dendrites_states);
        time_step_single_neuron<<<1,1>>>(n2.state, n2.decay_rate, n2.n_dendrites, n2.dendrites_states);
        cout << n1 << n2;
    }

    return 0;
}