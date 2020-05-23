
#include "main.hu"

using namespace std;

Neuron::Neuron (string _name) {
    name = new string;
    *name = _name;
    state = new float;
    *state = 0.;
    decay_rate = new float;
    *decay_rate = 10.;
    n_dendrites = new int;
    *n_dendrites = 0;
    dendrites = new Connection* [MAX_DENDRITES];
}
Neuron::~Neuron () {
    for (int i=0; i<*n_dendrites; i++) {
        delete dendrites[i];
    }
    delete[] dendrites;
    delete name, state, decay_rate, n_dendrites;
}

void Neuron::time_step () {
    // decay previous state
    *state *= exp(-1./(*decay_rate));
    // read from the dendrites
    for (int i=0; i<(*n_dendrites); i++) {
        adjust_state(dendrites[i]->time_step());
    }
}

void Neuron::adjust_state (float dx) {
    *state += dx;
}

float Neuron::get_state () {
    // use sigmoid activation
    return 1./(1.+exp(-*state));
}

void Neuron::attach_dendrite (Neuron* neuron) {
    dendrites[*n_dendrites] = new Connection(neuron);
    (*n_dendrites)++;
}

ostream& operator<< (ostream& cout, const Neuron& n) {
    cout << "Neuron" << *(n.name) << ": state=" << *(n.state) << ", decay_rate=" << *(n.decay_rate) << ", n_dendrites=" << *(n.n_dendrites) << endl;
    for (int i=0; i<*(n.n_dendrites); i++) {
        cout << *(n.dendrites[i]);
    }
    return cout;
}