main: main.o connection.o neuron.o brain.o
	nvcc -o main main.o connection.o neuron.o brain.o

%.o: %.cu main.hu
	nvcc -dc $<

PHONY: clean
clean:
	rm *.o

PHONY: cleanall
cleanall:
	rm *.o main