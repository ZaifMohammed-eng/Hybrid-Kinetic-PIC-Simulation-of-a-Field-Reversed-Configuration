This document describes the physics content, numerical implementation, and practical operation
of frc hybrid pic.py, a hybrid-kinetic particle-in-cell (PIC) simulation of a field-reversed
configuration (FRC) plasma built on WarpX 26.05 through the PICMI interface, using WarpX’s
Ohm’s-law hybrid solver. The code targets the kinetic regime of compact, high-density FRCs
(n ∼ 1022 m−3, Ti ∼ keV, rs ∼ 2 cm), where the number of ion gyroradii between the field null and
the separatrix is of order unity and finite-Larmor-radius (FLR) physics governs global stability.
The document is organized as follows. Section 2 explains why a hybrid model is the appropriate
tool and quantifies the scale separation that makes full explicit PIC intractable. Section 3 states
the governing equations exactly as the solver implements them. Section 4 derives the rigid-rotor
FRC equilibrium loaded at t = 0 and maps each formula to the code. Section 5 covers the
derived numerical constraints (grid, timestep, sub-stepping) and the stability conditions behind
them. Section 6 documents the Coulomb and D–D fusion modules, including the nuclear-mass
bookkeeping that WarpX enforces. Section 7 describes the physics diagnostics. Section 8 is a
tuning guide: every adjustable parameter, what it controls physically, and how to move it for
better results. Section 9 states the model’s limitations honestly. Section 10 lays out what must
be added—physics and software—to run credible 3D simulations of real FRC devices on HPC
resources.
