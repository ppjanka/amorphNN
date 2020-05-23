#include <iostream>
#include "main.hu"

using namespace std;

int main (void) {

    Neuron n1 ("1"), n2 ("2");

    cout << n1 << n2;
    n2.attach_dendrite(&n1);
    n1.adjust_state(1.);
    cout << n1 << n2;

    for (int time=0; time<10; time++) {
        n1.time_step();
        n2.time_step();
        cout << n1 << n2;
    }

    return 0;
}