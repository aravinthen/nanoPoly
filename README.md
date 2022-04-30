# nanoPoly

`nanoPoly` is a suite of tools that allow the design and coarse-grained atomistic simulation of a range of polymer architectures. 


The following structures are all supported:
* Polymer melts.
* Block copolymers of any conceivable configuration, as well as the melts composed by such chains.
* Branched polymers of any complexity.
* Chemically cross-linked rubber.

The structures can then be simulated with the ful capacity of the `LAMMPS` molecular dynamics engine.

An important feature of polymeric materials is the manner in some of them self-assemble into ordered mesoscopic structures. This is an extremely difficult process to simulate with molecular dynamics alone. `nanoPoly` allows for the use of density-biased self-avoiding random walks to generate ordered mesostructures whilst preserving the randomness of the microstructure.

## Dependencies
* LAMMPS
* `pscf` and `pscfFieldGen` for block copolymer morphology prediction.