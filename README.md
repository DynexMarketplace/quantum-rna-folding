# Quantum RNA Folding
In biology and chemistry, the properties of a molecule are not solely determined by a set of atoms but also by the shape of the molecule. In genetics, the shape of an RNA molecule is largely determined by how it bends back on itself. The sequence of A’s, U’s, G’s, and C’s that make up RNA has certain pairs that are drawn together to form hydrogen bonds. A sequence of several bonds in a row is called a stem, and a stem provides sufficient force to keep the molecule folded together. RNA molecules naturally form some stems while avoiding others in a manner that minimizes the free energy of the system. This demo program takes an RNA sequence and applies a quadratic model in pursuit of the optimal stem configuration.

# Usage

1. Launch in Github Codespaces and wait until the codepsace is fully initialised

2. Add your account keys by drag&drop of your dynex.ini into the main folder

3. Verify your account keys by typing the following command in the console:

```
python
>>> import dynex
>>> dynex.test()
>>> exit()
```

Your console will perform tests and validate your account keys. You should see the following message:

```
[DYNEX] TEST RESULT: ALL TESTS PASSED
```

4. Run the demo by typing the following command:

```
python main.py
```

The program will output and save the optimal grouping of the satellites into constellations in the file "result.png".


