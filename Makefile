main: main.o connection.o neuron.o
	nvcc -o main main.o connection.o neuron.o

%.o: %.cu
	nvcc -c $<

PHONY: clean
clean:
	rm *.o

PHONY: cleanall
cleanall:
	rm *.o main